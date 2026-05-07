use std::fs;
use std::io::BufReader;

use rand_core::{OsRng, RngCore};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::errors::ResultBoxedError;
use crate::utils::format::*;
use crate::utils::matrices::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Database {
  entries: Vec<u32>,
  m: usize,
  width: usize,
  elem_size: usize,
  plaintext_bits: usize,
}

impl Database {
  pub fn new(
    elements: &[String],
    m: usize,
    elem_size: usize,
    plaintext_bits: usize,
  ) -> ResultBoxedError<Self> {
    let rows = construct_rows(elements, m, elem_size, plaintext_bits)?;
    let width = if rows.is_empty() { 0 } else { rows[0].len() };

    Ok(Self {
      entries: flatten_matrix_col_major(&rows),
      m,
      width,
      elem_size,
      plaintext_bits,
    })
  }

  pub fn from_file(
    db_file: &str,
    m: usize,
    elem_size: usize,
    plaintext_bits: usize,
  ) -> ResultBoxedError<Self> {
    let file_contents: String = fs::read_to_string(db_file)?.parse()?;
    let elements: Vec<String> = serde_json::from_str(&file_contents)?;
    Self::new(&elements, m, elem_size, plaintext_bits)
  }

  pub fn vec_mult(&self, row: &[u32], col_idx: usize) -> u32 {
    let start = col_idx * self.m;
    let col = &self.entries[start..start + self.m];
    vec_mult_u32_u32(row, col).unwrap()
  }

  /// Generic batched matrix multiplication for N interleaved queries.
  /// Because N is a compile-time constant, LLVM will hopefully unroll
  /// the inner loop and auto-vectorize it to AVX/NEON instructions.
  pub fn vec_mult_batched_n<const N: usize>(
    &self,
    q_interleaved: &[[u32; N]],
    col_idx: usize,
  ) -> [u32; N] {
    // Extract the contiguous column slice
    let start = col_idx * self.m;
    let col = &self.entries[start..start + self.m];
    let len = col.len();

    // By asserting equal lengths upfront, we prove to the compiler that
    // bounds checking inside the loop is unnecessary. LLVM will strip
    // the safety checks and unroll this into pure AVX/NEON instructions.
    assert_eq!(q_interleaved.len(), len);

    let mut acc = [0u32; N];

    for i in 0..len {
      let c = col[i];
      let q = q_interleaved[i];

      // Hopefully LLVM auto translates this to vector multiply-add
      for j in 0..N {
        acc[j] = acc[j].wrapping_add(q[j].wrapping_mul(c));
      }
    }

    acc
  }

  pub fn write_to_file(&self, path: &str) -> ResultBoxedError<()> {
    let mut entries_2d = Vec::with_capacity(self.width);
    for col in 0..self.width {
      let start = col * self.m;
      entries_2d.push(self.entries[start..start + self.m].to_vec());
    }
    let json = json!(entries_2d);
    Ok(serde_json::to_writer(&fs::File::create(path)?, &json)?)
  }

  /// Returns the ith row of the DB matrix
  pub fn get_row(&self, i: usize) -> Vec<u32> {
    let mut row = Vec::with_capacity(self.width);
    for col in 0..self.width {
      row.push(self.entries[col * self.m + i]);
    }
    row
  }

  /// Returns the ith DB entry as a base64-encoded string
  pub fn get_db_entry(&self, i: usize) -> String {
    base64_from_u32_slice(&self.get_row(i), self.plaintext_bits, self.elem_size)
  }

  /// Returns the width of the DB matrix
  pub fn get_matrix_width(element_size: usize, plaintext_bits: usize) -> usize {
    let mut quo = element_size / plaintext_bits;
    if element_size % plaintext_bits != 0 {
      quo += 1;
    }
    quo
  }

  /// Returns the width of the DB matrix
  pub fn get_matrix_width_self(&self) -> usize {
    self.width
  }

  /// Get the matrix size
  pub fn get_matrix_height(&self) -> usize {
    self.m
  }

  /// Get the element size
  pub fn get_elem_size(&self) -> usize {
    self.elem_size
  }

  /// Get the plaintext bits
  pub fn get_plaintext_bits(&self) -> usize {
    self.plaintext_bits
  }
}

/// The `BaseParams` object allows loading and interacting with params that
/// are used by the client for constructing queries
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BaseParams {
  dim: usize, // the lwe dimension

  m: usize,         // the size of the DB
  elem_size: usize, // the size (in bits) of each element of the DB. Corresponds to `w` in paper.
  plaintext_bits: usize,

  public_seed: [u8; 32],
  rhs: Vec<u32>,
}

impl BaseParams {
  pub fn new(db: &Database, dim: usize) -> Self {
    let public_seed = generate_seed(); // generates the public seed
    Self {
      public_seed,
      rhs: Self::generate_params_rhs(db, public_seed, dim, db.m),
      dim,
      m: db.m,
      elem_size: db.elem_size,
      plaintext_bits: db.plaintext_bits,
    }
  }

  /// Load params from a JSON file
  pub fn load(params_path: &str) -> ResultBoxedError<Self> {
    let reader = BufReader::new(fs::File::open(params_path)?);
    Ok(serde_json::from_reader(reader)?)
  }

  /// Generates the RHS of the params using the database and the seed
  /// for the LHS
  pub fn generate_params_rhs(
    db: &Database,
    public_seed: [u8; 32],
    dim: usize,
    m: usize,
  ) -> Vec<u32> {
    let lhs =
      swap_matrix_fmt(&generate_lwe_matrix_from_seed(public_seed, dim, m));

    (0..db.get_matrix_width_self())
      .into_par_iter()
      .flat_map_iter(|i| {
        let mut col = Vec::with_capacity(dim);
        for r in &lhs {
          col.push(db.vec_mult(r, i));
        }
        col
      })
      .collect()
  }

  /// Writes the params struct as JSON to file
  pub fn write_to_file(&self, path: &str) -> ResultBoxedError<()> {
    let json = json!({
      "public_seed": self.public_seed,
      "rhs": self.rhs,
    });
    Ok(serde_json::to_writer(&fs::File::create(path)?, &json)?)
  }

  /// Computes c = s*(A*DB) using the RHS of the public parameters
  pub fn mult_right(&self, s: &[u32]) -> ResultBoxedError<Vec<u32>> {
    let width = self.rhs.len() / self.dim;
    let res: Vec<u32> = (0..width)
      .into_par_iter()
      .map(|i| {
        let start = i * self.dim;
        let col = &self.rhs[start..start + self.dim];
        vec_mult_u32_u32(s, col).unwrap()
      })
      .collect();
    Ok(res)
  }

  pub fn mult_right_batched_n<const N: usize>(
    &self,
    s_interleaved: &[[u32; N]],
  ) -> ResultBoxedError<Vec<[u32; N]>> {
    let width = self.rhs.len() / self.dim;
    let res: Vec<[u32; N]> = (0..width)
      .into_par_iter()
      .map(|col_idx| {
        let start = col_idx * self.dim;
        let col = &self.rhs[start..start + self.dim];
        assert_eq!(s_interleaved.len(), self.dim);

        let mut acc = [0u32; N];
        for i in 0..self.dim {
          let c = col[i];
          let s = s_interleaved[i];
          for j in 0..N {
            acc[j] = acc[j].wrapping_add(s[j].wrapping_mul(c));
          }
        }
        acc
      })
      .collect();
    Ok(res)
  }

  pub fn get_total_records(&self) -> usize {
    self.m
  }

  pub fn get_dim(&self) -> usize {
    self.dim
  }

  pub fn get_elem_size(&self) -> usize {
    self.elem_size
  }

  pub fn get_plaintext_bits(&self) -> usize {
    self.plaintext_bits
  }
}

/// `CommonParams` holds the derived uniform matrix that is used for
/// constructing the server's public parameters and the client query.
#[derive(Serialize, Deserialize)]
pub struct CommonParams {
  matrix: Vec<u32>,
  m: usize,
  dim: usize,
}

impl CommonParams {
  // Returns the internal matrix
  pub fn as_matrix_flat(&self) -> &[u32] {
    &self.matrix
  }

  /// Computes b = s*A + e using the seed used to generate the matrix of
  /// the public parameters
  pub fn mult_left(&self, s: &[u32]) -> ResultBoxedError<Vec<u32>> {
    let res: Vec<u32> = (0..self.m)
      .into_par_iter()
      .map(|i| {
        let start = i * self.dim;
        let col = &self.matrix[start..start + self.dim];
        let s_a = vec_mult_u32_u32(s, col).unwrap();
        let e = random_ternary();
        s_a.wrapping_add(e)
      })
      .collect();
    Ok(res)
  }

  pub fn mult_left_batched_n<const N: usize>(
    &self,
    s_interleaved: &[[u32; N]],
  ) -> ResultBoxedError<Vec<[u32; N]>> {
    let res: Vec<[u32; N]> = (0..self.m)
      .into_par_iter()
      .map(|col_idx| {
        let start = col_idx * self.dim;
        let col = &self.matrix[start..start + self.dim];
        assert_eq!(s_interleaved.len(), self.dim);

        let mut acc = [0u32; N];
        for i in 0..self.dim {
          let c = col[i];
          let s = s_interleaved[i];
          for j in 0..N {
            acc[j] = acc[j].wrapping_add(s[j].wrapping_mul(c));
          }
        }
        // Add random ternary error for each batched element
        for j in 0..N {
          acc[j] = acc[j].wrapping_add(random_ternary());
        }
        acc
      })
      .collect();
    Ok(res)
  }
}

impl From<&BaseParams> for CommonParams {
  fn from(params: &BaseParams) -> Self {
    let cols =
      generate_lwe_matrix_from_seed(params.public_seed, params.dim, params.m);
    Self {
      matrix: cols.into_iter().flatten().collect(),
      m: params.m,
      dim: params.dim,
    }
  }
}

fn construct_rows(
  elements: &[String],
  m: usize,
  elem_size: usize,
  plaintext_bits: usize,
) -> ResultBoxedError<Vec<Vec<u32>>> {
  let row_width = Database::get_matrix_width(elem_size, plaintext_bits);

  let result = (0..m)
    .into_par_iter()
    .map(|i| -> ResultBoxedError<Vec<u32>> {
      let mut row = Vec::with_capacity(row_width);
      let data = &elements[i];
      let bytes = base64::decode(data)?;
      let bits = bytes_to_bits_le(&bytes);
      for i in 0..row_width {
        let end_bound = (i + 1) * plaintext_bits;
        if end_bound < bits.len() {
          row.push(bits_to_u32_le(&bits[i * plaintext_bits..end_bound])?);
        } else {
          row.push(bits_to_u32_le(&bits[i * plaintext_bits..])?);
        }
      }
      Ok(row)
    });

  result.collect()
}

fn generate_seed() -> [u8; 32] {
  let mut seed = [0u8; 32];
  OsRng.fill_bytes(&mut seed);
  seed
}
