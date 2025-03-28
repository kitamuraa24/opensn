// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "framework/math/math.h"

namespace opensn
{

/**
 * Sparse matrix utility. This is a basic CSR type sparse matrix which allows efficient matrix
 * storage and multiplication. It is not intended for solving linear systems (use PETSc for that
 * instead). It was originally developed for the transfer matrices of transport cross sections.
 */
class SparseMatrix
{
private:
  /// Maximum number of rows for this matrix
  size_t row_size_;
  /// Maximum number of columns for this matrix
  size_t col_size_;

public:
  /// rowI_indices[i] is a vector indices j for the non-zero columns.
  std::vector<std::vector<size_t>> rowI_indices;
  /// rowI_values[i] corresponds to column indices and contains the non-zero value.
  std::vector<std::vector<double>> rowI_values;

public:
  /// Constructor with number of rows and columns constructor.
  SparseMatrix(size_t num_rows, size_t num_cols);
  /// Copy constructor.
  SparseMatrix(const SparseMatrix& matrix);

  size_t GetNumRows() const { return row_size_; }
  size_t GetNumCols() const { return col_size_; }

  /// Inserts a value into the matrix.
  void Insert(size_t i, size_t j, double value);

  /// Inserts-Adds a value into the matrix with duplicate check.
  void InsertAdd(size_t i, size_t j, double value);

  /**
   * Returns the value in the matrix at the given location. This is a rather inefficient routine.
   * Use the columns and values rather than directly this function.
   */
  double GetValueIJ(size_t i, size_t j) const;

  /// Sets the diagonal of the matrix using a vector.
  void SetDiagonal(const std::vector<double>& diag);

  /// Sorts the column indices of each row for faster lookup.
  void Compress();

  /// Prints the sparse matrix to string.
  std::string PrintStr() const;

private:
  /// Constructor with number of rows constructor.
  void CheckInitialized() const;

public:
  virtual ~SparseMatrix() = default;

public:
  struct EntryReference
  {
    const size_t& row_index;
    const size_t& column_index;
    double& value;

    EntryReference(const size_t& row_id, const size_t& column_id, double& value)
      : row_index(row_id), column_index(column_id), value(value)
    {
    }
  };

  struct ConstEntryReference
  {
  public:
    const size_t& row_index;
    const size_t& column_index;
    const double& value;

    ConstEntryReference(const size_t& row_id, const size_t& column_id, const double& value)
      : row_index(row_id), column_index(column_id), value(value)
    {
    }
  };

  class RowIteratorContext
  {
  private:
    const std::vector<size_t>& ref_col_ids_;
    std::vector<double>& ref_col_vals_;
    const size_t ref_row_;

  public:
    RowIteratorContext(SparseMatrix& matrix, size_t ref_row)
      : ref_col_ids_(matrix.rowI_indices[ref_row]),
        ref_col_vals_(matrix.rowI_values[ref_row]),
        ref_row_(ref_row)
    {
    }

    class RowIterator
    {
    private:
      RowIteratorContext& context_;
      size_t ref_entry_;

    public:
      RowIterator(RowIteratorContext& context, size_t ref_entry)
        : context_{context}, ref_entry_{ref_entry}
      {
      }

      RowIterator operator++()
      {
        RowIterator i = *this;
        ref_entry_++;
        return i;
      }
      RowIterator operator++(int)
      {
        ref_entry_++;
        return *this;
      }

      EntryReference operator*()
      {
        return {
          context_.ref_row_, context_.ref_col_ids_[ref_entry_], context_.ref_col_vals_[ref_entry_]};
      }

      bool operator==(const RowIterator& rhs) const { return ref_entry_ == rhs.ref_entry_; }
      bool operator!=(const RowIterator& rhs) const { return ref_entry_ != rhs.ref_entry_; }
    };

    RowIterator begin() { return {*this, 0}; }
    RowIterator end() { return {*this, ref_col_vals_.size()}; }
  };

  RowIteratorContext Row(size_t row_id);

  class ConstRowIteratorContext
  {
  private:
    const std::vector<size_t>& ref_col_ids_;
    const std::vector<double>& ref_col_vals_;
    const size_t ref_row_;

  public:
    ConstRowIteratorContext(const SparseMatrix& matrix, size_t ref_row)
      : ref_col_ids_(matrix.rowI_indices[ref_row]),
        ref_col_vals_(matrix.rowI_values[ref_row]),
        ref_row_(ref_row)
    {
    }

    class ConstRowIterator
    {
    private:
      const ConstRowIteratorContext& context_;
      size_t ref_entry_;

    public:
      ConstRowIterator(const ConstRowIteratorContext& context, size_t ref_entry)
        : context_(context), ref_entry_{ref_entry}
      {
      }

      ConstRowIterator operator++()
      {
        ConstRowIterator i = *this;
        ref_entry_++;
        return i;
      }
      ConstRowIterator operator++(int)
      {
        ref_entry_++;
        return *this;
      }

      ConstEntryReference operator*()
      {
        return {
          context_.ref_row_, context_.ref_col_ids_[ref_entry_], context_.ref_col_vals_[ref_entry_]};
      }

      bool operator==(const ConstRowIterator& rhs) const { return ref_entry_ == rhs.ref_entry_; }
      bool operator!=(const ConstRowIterator& rhs) const { return ref_entry_ != rhs.ref_entry_; }
    };

    ConstRowIterator begin() const { return {*this, 0}; }
    ConstRowIterator end() const { return {*this, ref_col_vals_.size()}; }
  };

  ConstRowIteratorContext Row(size_t row_id) const;

  /// Iterator to loop over all matrix entries.
  class EntriesIterator
  {
  private:
    SparseMatrix& sp_matrix;
    size_t ref_row_;
    size_t ref_col_;

  public:
    explicit EntriesIterator(SparseMatrix& context, size_t row)
      : sp_matrix{context}, ref_row_{row}, ref_col_(0)
    {
    }

    void Advance()
    {
      ref_col_++;
      if (ref_col_ >= sp_matrix.rowI_indices[ref_row_].size())
      {
        ref_row_++;
        ref_col_ = 0;
        while ((ref_row_ < sp_matrix.row_size_) and (sp_matrix.rowI_indices[ref_row_].empty()))
          ref_row_++;
      }
    }

    EntriesIterator operator++()
    {
      EntriesIterator i = *this;
      Advance();
      return i;
    }
    EntriesIterator operator++(int)
    {
      Advance();
      return *this;
    }

    EntryReference operator*()
    {
      return {ref_row_,
              sp_matrix.rowI_indices[ref_row_][ref_col_],
              sp_matrix.rowI_values[ref_row_][ref_col_]};
    }
    bool operator==(const EntriesIterator& rhs) const
    {
      return (ref_row_ == rhs.ref_row_) and (ref_col_ == rhs.ref_col_);
    }
    bool operator!=(const EntriesIterator& rhs) const
    {
      return (ref_row_ != rhs.ref_row_) or (ref_col_ != rhs.ref_col_);
    }
  };

  EntriesIterator begin();
  EntriesIterator end();
};

} // namespace opensn
