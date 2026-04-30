/// The `api` module is the public entry point for all FrodoPIR database.
use crate::db::Database;
pub use crate::db::{BaseParams, CommonParams};
use crate::errors::{
  ErrorOverflownAdd, ErrorQueryParamsReused, ResultBoxedError,
};
pub use crate::utils::format::*;
use crate::utils::lwe::*;
use crate::utils::matrices::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::str;

use rayon::prelude::*;

/// A `Shard` is an instance of a database, where each row corresponds
/// to a single element, that has been preprocessed by the server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Shard {
  db: Database,
  base_params: BaseParams,
}

impl Shard {
  /// Expects a JSON file of base64-encoded strings in file path. It also
  /// expects the lwe dimension, m (the number of DB elements), element size
  /// (in bytes) of the database elements, and plaintext bits.
  /// It will call the 'from_base64_strings' function to generate the database.
  pub fn from_json_file(
    file_path: &str,
    lwe_dim: usize,
    m: usize,
    elem_size: usize,
    plaintext_bits: usize,
  ) -> ResultBoxedError<Self> {
    let file_contents: String =
      fs::read_to_string(file_path).unwrap().parse().unwrap();
    let elements: Vec<String> = serde_json::from_str(&file_contents).unwrap();
    Shard::from_base64_strings(&elements, lwe_dim, m, elem_size, plaintext_bits)
  }

  /// Expects an array of base64-encoded strings and converts into a
  /// database that can process client queries
  pub fn from_base64_strings(
    base64_strs: &[String],
    lwe_dim: usize,
    m: usize,
    elem_size: usize,
    plaintext_bits: usize,
  ) -> ResultBoxedError<Self> {
    let db = Database::new(base64_strs, m, elem_size, plaintext_bits)?;
    let base_params = BaseParams::new(&db, lwe_dim);
    Ok(Self { db, base_params })
  }

  /// Write base_params and DB to file
  pub fn write_to_file(
    &self,
    db_path: &str,
    params_path: &str,
  ) -> ResultBoxedError<()> {
    self.db.write_to_file(db_path)?;
    self.base_params.write_to_file(params_path)?;
    Ok(())
  }

  // Produces a serialized response (base64-encoded) to a serialized
  // client query: c' = b' * DB
  pub fn respond(&self, q: &Query) -> ResultBoxedError<Vec<u8>> {
    let q = q.as_slice();
    let resp = Response(
      (0..self.db.get_matrix_width_self())
        .map(|i| self.db.vec_mult(q, i))
        .collect(),
    );
    let ser = bincode::serialize(&resp);

    Ok(ser?)
  }

  /// Returns the database
  pub fn get_db(&self) -> &Database {
    &self.db
  }

  /// Returns the base parameters
  pub fn get_base_params(&self) -> &BaseParams {
    &self.base_params
  }

  pub fn into_row_iter(&self) -> std::vec::IntoIter<std::string::String> {
    (0..self.get_db().get_matrix_height())
      .map(|i| self.get_db().get_db_entry(i))
      .collect::<Vec<String>>()
      .into_iter()
  }
}

/// The `QueryParams` struct is initialized to be used for a client
/// query.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryParams {
  lhs: Vec<u32>,
  rhs: Vec<u32>,
  elem_size: usize,
  plaintext_bits: usize,
  pub used: bool,
}

impl QueryParams {
  pub fn new(cp: &CommonParams, bp: &BaseParams) -> ResultBoxedError<Self> {
    let s = random_ternary_vector(bp.get_dim()); // The `s` value for the client as in the paper
    Ok(Self {
      lhs: cp.mult_left(&s)?,  // The `b` value
      rhs: bp.mult_right(&s)?, // The `c` value
      elem_size: bp.get_elem_size(),
      plaintext_bits: bp.get_plaintext_bits(),
      used: false,
    })
  }

  /// Prepares a new client query based on an input row_index
  pub fn generate_query(
    &mut self,
    row_index: usize,
  ) -> ResultBoxedError<Query> {
    if self.used {
      return Err(Box::new(ErrorQueryParamsReused {}));
    }
    self.used = true;
    let query_indicator = get_rounding_factor(self.plaintext_bits);
    let mut lhs = Vec::new();
    lhs.clone_from(&self.lhs.clone());
    let (result, check) = lhs[row_index].overflowing_add(query_indicator);
    if !check {
      lhs[row_index] = result;
    } else {
      return Err(Box::new(ErrorOverflownAdd {}));
    }
    Ok(Query(lhs))
  }
}

/// The `Query` struct holds the necessary information encoded in
/// a client PIR query to the server DB for a particular `row_index`. It
/// provides methods for parsing server responses.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Query(Vec<u32>);
impl Query {
  pub fn as_slice(&self) -> &[u32] {
    &self.0
  }
}

/// The `Response` object wraps a response from a single shard
#[derive(Clone, Serialize, Deserialize)]
pub struct Response(Vec<u32>);
impl Response {
  pub fn as_slice(&self) -> &[u32] {
    &self.0
  }

  /// Parses the output as a row of u32 values
  pub fn parse_output_as_row(&self, qp: &QueryParams) -> Vec<u32> {
    // get parameters for rounding
    let rounding_factor = get_rounding_factor(qp.plaintext_bits);
    let rounding_floor = get_rounding_floor(qp.plaintext_bits);
    let plaintext_size = get_plaintext_size(qp.plaintext_bits);

    // perform division and rounding in parallel
    (0..Database::get_matrix_width(qp.elem_size, qp.plaintext_bits))
      .into_par_iter()
      .map(|i| {
        let unscaled_res = self.0[i].wrapping_sub(qp.rhs[i]);
        let scaled_res = unscaled_res / rounding_factor;
        let scaled_rem = unscaled_res % rounding_factor;
        let mut rounded_res = scaled_res;
        if scaled_rem > rounding_floor {
          rounded_res += 1;
        }
        rounded_res % plaintext_size
      })
      .collect()
  }

  /// Parses the output as bytes
  pub fn parse_output_as_bytes(&self, qp: &QueryParams) -> Vec<u8> {
    let row = self.parse_output_as_row(qp);
    bytes_from_u32_slice(&row, qp.plaintext_bits, qp.elem_size)
  }

  /// Parses the output as a base64-encoded string
  pub fn parse_output_as_base64(&self, qp: &QueryParams) -> String {
    let row = self.parse_output_as_row(qp);
    base64_from_u32_slice(&row, qp.plaintext_bits, qp.elem_size)
  }
}

/// Macro to generate structs and method for generic batching
/// The `BatchedQuery` struct holds interleaved client PIR queries.
/// The `BatchedResponse` object wraps the responses for client queries.
/// Produces a serialized response to interleaved queries simultaneously.
macro_rules! impl_batched_api {
  ($n:expr, $query_name:ident, $resp_name:ident, $method_name:ident) => {
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct $query_name {
      pub interleaved: Vec<[u32; $n]>,
    }

    #[derive(Clone, Serialize, Deserialize)]
    pub struct $resp_name {
      pub results: Vec<[u32; $n]>,
    }

    impl Shard {
      pub fn $method_name(
        &self,
        bq: &$query_name,
      ) -> ResultBoxedError<Vec<u8>> {
        let width = self.db.get_matrix_width_self();

        // Par iterator hopefully splits the columns
        let results: Vec<[u32; $n]> = (0..width)
          .into_par_iter()
          .map(|i| self.db.vec_mult_batched_n::<$n>(&bq.interleaved, i))
          .collect();

        let resp = $resp_name { results };
        let ser = bincode::serialize(&resp);
        Ok(ser?)
      }
    }
  };
}

impl_batched_api!(2, BatchedQuery2, BatchedResponse2, respond_batched_2);
impl_batched_api!(4, BatchedQuery4, BatchedResponse4, respond_batched_4);
impl_batched_api!(8, BatchedQuery8, BatchedResponse8, respond_batched_8);
impl_batched_api!(16, BatchedQuery16, BatchedResponse16, respond_batched_16);
impl_batched_api!(32, BatchedQuery32, BatchedResponse32, respond_batched_32);

#[cfg(test)]
mod tests {
  use super::*;
  use rand_core::{OsRng, RngCore};

  #[test]
  fn client_query_to_server_10_times() {
    let m = 2u32.pow(12) as usize;
    let elem_size = 2u32.pow(8) as usize;
    let plaintext_bits = 12usize;
    let lwe_dim = 512;
    let db_elems = generate_db_elems(m, (elem_size + 7) / 8);
    let shard = Shard::from_base64_strings(
      &db_elems,
      lwe_dim,
      m,
      elem_size,
      plaintext_bits,
    )
    .unwrap();

    let bp = shard.get_base_params();
    let cp = CommonParams::from(bp);

    #[allow(clippy::needless_range_loop)]
    for i in 0..10 {
      let mut qp = QueryParams::new(&cp, bp).unwrap();
      let q = qp.generate_query(i).unwrap();

      let d_resp = shard.respond(&q).unwrap();
      let resp: Response = bincode::deserialize(&d_resp).unwrap();

      let output = resp.parse_output_as_base64(&qp);
      assert_eq!(output, db_elems[i]);
    }
  }

  #[test]
  fn client_query_to_server_attempt_params_reuse() {
    let m = 2u32.pow(6) as usize;
    let elem_size = 2u32.pow(8) as usize;
    let plaintext_bits = 10usize;
    let lwe_dim = 512;
    let db_elems = generate_db_elems(m, (elem_size + 7) / 8);
    let shard = Shard::from_base64_strings(
      &db_elems,
      lwe_dim,
      m,
      elem_size,
      plaintext_bits,
    )
    .unwrap();
    let bp = shard.get_base_params();
    let cp = CommonParams::from(bp);

    let mut qp = QueryParams::new(&cp, bp).unwrap();

    // should be successful in generating a query
    let res_unused = qp.generate_query(0);
    assert!(res_unused.is_ok());

    // should be "used"
    assert!(qp.used);

    // should be successful in generating a query
    let res = qp.generate_query(0);
    assert!(res.is_err());
  }

  // This will generate random elements for test databases
  fn generate_db_elems(num_elems: usize, elem_byte_len: usize) -> Vec<String> {
    let mut elems = Vec::with_capacity(num_elems);
    for _ in 0..num_elems {
      let mut elem = vec![0u8; elem_byte_len];
      OsRng.fill_bytes(&mut elem);
      let elem_str = base64::encode(elem);
      elems.push(elem_str);
    }
    elems
  }

  macro_rules! test_batch_size {
    ($test_name:ident, $n:expr, $query_type:ident, $resp_type:ident, $method:ident) => {
      #[test]
      fn $test_name() {
        use std::time::Instant;

        let m = 2u32.pow(18) as usize;
        let elem_size = 2u32.pow(13) as usize;
        let plaintext_bits = 10usize;
        let lwe_dim = 1572;

        let db_elems = generate_db_elems(m, (elem_size + 7) / 8);
        let shard = Shard::from_base64_strings(
          &db_elems,
          lwe_dim,
          m,
          elem_size,
          plaintext_bits,
        )
        .unwrap();

        let bp = shard.get_base_params();
        let cp = CommonParams::from(bp);

        let mut qps: Vec<QueryParams> = (0..$n)
          .map(|_| QueryParams::new(&cp, bp).unwrap())
          .collect();

        let mut queries = Vec::with_capacity($n);
        let mut target_indices = Vec::with_capacity($n);

        for i in 0..$n {
          let idx = (i * (m / $n)) % m;
          target_indices.push(idx);
          queries.push(qps[i].generate_query(idx).unwrap());
        }

        let interleaved: Vec<[u32; $n]> = (0..m)
          .into_par_iter()
          .map(|row| {
            let mut vals = [0u32; $n];
            for j in 0..$n {
              vals[j] = queries[j].as_slice()[row];
            }
            vals
          })
          .collect();

        let bq = super::$query_type { interleaved };

        let iterations = 10;

        let _ = shard.respond(&queries[0]).unwrap();
        let _ = shard.$method(&bq).unwrap();

        let start_seq = Instant::now();
        for _ in 0..iterations {
          for i in 0..$n {
            let _ = shard.respond(&queries[i]).unwrap();
          }
        }
        let duration_seq = start_seq.elapsed();

        let start_batched = Instant::now();
        for _ in 0..iterations {
          let _ = shard.$method(&bq).unwrap();
        }
        let duration_batched = start_batched.elapsed();

        println!("\n--------------------------------------------------");
        println!(
          "Performance Results for Batch Size {} ({} iterations):",
          $n, iterations
        );
        println!("Sequential ({} separate requests): {:?}", $n, duration_seq);
        println!("Interleaved Batched (1 request):   {:?}", duration_batched);
        println!(
          "Speedup multiplier: {:.2}x",
          duration_seq.as_secs_f64() / duration_batched.as_secs_f64()
        );
        println!("--------------------------------------------------");

        let d_resp = shard.$method(&bq).unwrap();
        let resp: super::$resp_type = bincode::deserialize(&d_resp).unwrap();
        let width = shard.get_db().get_matrix_width_self();

        for j in 0..$n {
          let mut single_resp = Vec::with_capacity(width);
          for w in 0..width {
            single_resp.push(resp.results[w][j]);
          }
          let output = Response(single_resp).parse_output_as_base64(&qps[j]);
          assert_eq!(output, db_elems[target_indices[j]]);
        }
      }
    };
  }

  test_batch_size!(
    test_batched_queries_2,
    2,
    BatchedQuery2,
    BatchedResponse2,
    respond_batched_2
  );
  test_batch_size!(
    test_batched_queries_4,
    4,
    BatchedQuery4,
    BatchedResponse4,
    respond_batched_4
  );
  test_batch_size!(
    test_batched_queries_8,
    8,
    BatchedQuery8,
    BatchedResponse8,
    respond_batched_8
  );
  test_batch_size!(
    test_batched_queries_16,
    16,
    BatchedQuery16,
    BatchedResponse16,
    respond_batched_16
  );
  test_batch_size!(
    test_batched_queries_32,
    32,
    BatchedQuery32,
    BatchedResponse32,
    respond_batched_32
  );
}
