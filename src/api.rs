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

pub mod array_serde {
    use serde::{ser::{SerializeSeq, SerializeTuple}, de::{SeqAccess, Visitor}, Deserializer, Serializer, Deserialize};
    use std::fmt;

    pub fn serialize<S, const N: usize>(arrays: &Vec<[u32; N]>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(arrays.len()))?;
        for array in arrays {
            struct ArrayTuple<'a, const N: usize>(&'a [u32; N]);
            impl<'a, const N: usize> serde::Serialize for ArrayTuple<'a, N> {
                fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
                where S: Serializer {
                    let mut tup = serializer.serialize_tuple(N)?;
                    for item in self.0 {
                        tup.serialize_element(item)?;
                    }
                    tup.end()
                }
            }
            seq.serialize_element(&ArrayTuple(array))?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D, const N: usize>(deserializer: D) -> Result<Vec<[u32; N]>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct VecVisitor<const N: usize>;

        impl<'de, const N: usize> Visitor<'de> for VecVisitor<N> {
            type Value = Vec<[u32; N]>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence of arrays")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut vec = match seq.size_hint() {
                    Some(size) => Vec::with_capacity(size),
                    None => Vec::new(),
                };

                struct ArrayTuple<const N: usize>([u32; N]);
                impl<'de, const N: usize> Deserialize<'de> for ArrayTuple<N> {
                    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
                    where D: Deserializer<'de> {
                        struct ArrayVisitor<const N: usize>;
                        impl<'de, const N: usize> Visitor<'de> for ArrayVisitor<N> {
                            type Value = [u32; N];
                            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                                formatter.write_str("an array of length N")
                            }
                            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                            where A: SeqAccess<'de> {
                                let mut arr = [0u32; N];
                                for i in 0..N {
                                    arr[i] = seq.next_element()?
                                        .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                                }
                                Ok(arr)
                            }
                        }
                        deserializer.deserialize_tuple(N, ArrayVisitor::<N>).map(ArrayTuple)
                    }
                }

                while let Some(tup) = seq.next_element::<ArrayTuple<N>>()? {
                    vec.push(tup.0);
                }
                Ok(vec)
            }
        }

        deserializer.deserialize_seq(VecVisitor::<N>)
    }
}

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
    let resp = Response(self.db.eval_single(q));
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
pub struct Response(pub Vec<u32>);
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
  ($n:expr, $query_name:ident, $resp_name:ident, $method_name:ident, $db_eval_method:ident) => {
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct $query_name {
      #[serde(with = "crate::api::array_serde")]
      pub interleaved: Vec<[u32; $n]>,
    }

    #[derive(Clone, Serialize, Deserialize)]
    pub struct $resp_name {
      #[serde(with = "crate::api::array_serde")]
      pub results: Vec<[u32; $n]>,
    }

    impl Shard {
      pub fn $method_name(
        &self,
        bq: &$query_name,
      ) -> ResultBoxedError<Vec<u8>> {
        
        let results = self.db.$db_eval_method(&bq.interleaved);

        let resp = $resp_name { results };
        let ser = bincode::serialize(&resp);
        Ok(ser?)
      }
    }
  };
}

macro_rules! impl_batched_query_params {
  ($n:expr, $name:ident) => {
    pub struct $name {
      pub lhs: Vec<[u32; $n]>,
      pub rhs: Vec<[u32; $n]>,
      pub elem_size: usize,
      pub plaintext_bits: usize,
      pub used: [bool; $n],
    }

    impl $name {
      pub fn new(cp: &CommonParams, bp: &BaseParams) -> ResultBoxedError<Self> {
        let dim = bp.get_dim();

        let mut s_interleaved = vec![[0u32; $n]; dim];
        for j in 0..$n {
          let s = random_ternary_vector(dim);
          for i in 0..dim {
            s_interleaved[i][j] = s[i];
          }
        }

        let lhs = cp.mult_left_batched_n::<$n>(&s_interleaved)?;
        let rhs = bp.mult_right_batched_n::<$n>(&s_interleaved)?;

        Ok(Self {
          lhs,
          rhs,
          elem_size: bp.get_elem_size(),
          plaintext_bits: bp.get_plaintext_bits(),
          used: [false; $n],
        })
      }

      pub fn extract_query_params(
        &mut self,
        index: usize,
      ) -> ResultBoxedError<QueryParams> {
        if self.used[index] {
          return Err(Box::new(ErrorQueryParamsReused {}));
        }
        self.used[index] = true;

        // De-interleave the requested query constraints
        let extracted_lhs: Vec<u32> =
          self.lhs.iter().map(|row| row[index]).collect();
        let extracted_rhs: Vec<u32> =
          self.rhs.iter().map(|row| row[index]).collect();

        Ok(QueryParams {
          lhs: extracted_lhs,
          rhs: extracted_rhs,
          elem_size: self.elem_size,
          plaintext_bits: self.plaintext_bits,
          used: false,
        })
      }
    }
  };
}

impl_batched_query_params!(2, BatchedQueryParams2);
impl_batched_query_params!(4, BatchedQueryParams4);
impl_batched_query_params!(8, BatchedQueryParams8);
impl_batched_query_params!(16, BatchedQueryParams16);
impl_batched_query_params!(32, BatchedQueryParams32);
impl_batched_query_params!(64, BatchedQueryParams64);
impl_batched_query_params!(128, BatchedQueryParams128);
impl_batched_query_params!(256, BatchedQueryParams256);
impl_batched_query_params!(512, BatchedQueryParams512);
impl_batched_query_params!(1024, BatchedQueryParams1024);

impl_batched_api!(2, BatchedQuery2, BatchedResponse2, respond_batched_2, eval_batched_2);
impl_batched_api!(4, BatchedQuery4, BatchedResponse4, respond_batched_4, eval_batched_4);
impl_batched_api!(8, BatchedQuery8, BatchedResponse8, respond_batched_8, eval_batched_8);
impl_batched_api!(16, BatchedQuery16, BatchedResponse16, respond_batched_16, eval_batched_16);
impl_batched_api!(32, BatchedQuery32, BatchedResponse32, respond_batched_32, eval_batched_32);
impl_batched_api!(64, BatchedQuery64, BatchedResponse64, respond_batched_64, eval_batched_64);
impl_batched_api!(128, BatchedQuery128, BatchedResponse128, respond_batched_128, eval_batched_128);
impl_batched_api!(256, BatchedQuery256, BatchedResponse256, respond_batched_256, eval_batched_256);
impl_batched_api!(512, BatchedQuery512, BatchedResponse512, respond_batched_512, eval_batched_512);
impl_batched_api!(1024, BatchedQuery1024, BatchedResponse1024, respond_batched_1024, eval_batched_1024);

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

  macro_rules! test_client_batch_size {
    ($test_name:ident, $n:expr, $batch_type:ident) => {
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

        let iterations = 10;

        // Warmup
        let _ = QueryParams::new(&cp, bp).unwrap();
        let _ = super::$batch_type::new(&cp, bp).unwrap();

        let start_seq = Instant::now();
        for _ in 0..iterations {
          for _ in 0..$n {
            let _ = QueryParams::new(&cp, bp).unwrap();
          }
        }
        let duration_seq = start_seq.elapsed();

        let start_batched = Instant::now();
        for _ in 0..iterations {
          let _ = super::$batch_type::new(&cp, bp).unwrap();
        }
        let duration_batched = start_batched.elapsed();

        println!("\n--------------------------------------------------");
        println!(
          "Client Preprocessing Performance for Batch Size {} ({} iterations):",
          $n, iterations
        );
        println!("Sequential ({} separate creations): {:?}", $n, duration_seq);
        println!("SIMD Batched (1 batched creation):  {:?}", duration_batched);
        println!(
          "Speedup multiplier: {:.2}x",
          duration_seq.as_secs_f64() / duration_batched.as_secs_f64()
        );
        println!("--------------------------------------------------");

        // Ensure that the extracted queries still retrieve the correct items
        let mut batched_qps = super::$batch_type::new(&cp, bp).unwrap();
        for j in 0..$n {
          let mut qp = batched_qps.extract_query_params(j).unwrap();
          let target_row = j; // Targeted the j-th row for simplicity

          let q = qp.generate_query(target_row).unwrap();
          let d_resp = shard.respond(&q).unwrap();
          let resp: super::Response = bincode::deserialize(&d_resp).unwrap();

          let output = resp.parse_output_as_base64(&qp);
          assert_eq!(
            output, db_elems[target_row],
            "Batched query {} failed!",
            j
          );
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
  test_batch_size!(
    test_batched_queries_64,
    64,
    BatchedQuery64,
    BatchedResponse64,
    respond_batched_64
  );
  test_batch_size!(test_batched_queries_128,
    128,
    BatchedQuery128,
     BatchedResponse128,
     respond_batched_128
   );
  test_batch_size!(
    test_batched_queries_256,
    256,
    BatchedQuery256,
    BatchedResponse256,
    respond_batched_256
  );
  test_batch_size!(test_batched_queries_512,
    512,
     BatchedQuery512,
     BatchedResponse512,
     respond_batched_512
   );
  test_batch_size!(
    test_batched_queries_1024,
    1024,
    BatchedQuery1024,
    BatchedResponse1024,
     respond_batched_1024
   );

  test_client_batch_size!(test_client_batch_2, 2, BatchedQueryParams2);
  test_client_batch_size!(test_client_batch_4, 4, BatchedQueryParams4);
  test_client_batch_size!(test_client_batch_8, 8, BatchedQueryParams8);
  test_client_batch_size!(test_client_batch_16, 16, BatchedQueryParams16);
  test_client_batch_size!(test_client_batch_32, 32, BatchedQueryParams32);
  test_client_batch_size!(test_client_batch_64, 64, BatchedQueryParams64);
  test_client_batch_size!(test_client_batch_128, 128, BatchedQueryParams128);
  test_client_batch_size!(test_client_batch_256, 256, BatchedQueryParams256);
  test_client_batch_size!(test_client_batch_512, 512, BatchedQueryParams512);
  test_client_batch_size!(test_client_batch_1024, 1024, BatchedQueryParams1024);
}
