# Vectorium

This Python package provides a simple and flexible interface for creating, manipulating and querying vector databases. Vectorium makes it easy to add or remove vectors, compare them using various similarity metrics, and perform various aggregate operations.

### Features

- **Flexible Vector Database**: Store and manage your high dimensional vectors in an efficient manner.
- **Multiple Compare Functions**: Includes functions like cosine similarity, euclidean distance, dot product.
- **Various Aggregate Functions**: Supports aggregation of result vectors using mean, sum, max, min or no operation.
- **Vector Operations**: Add or remove vectors from your database, save or load your vector collection, and update your vector database as needed.

## Installation

You can install Vectorium using pip:

```shell
pip install vectorium
```
## Usage

This is a brief example of how to use the `VectorDatabase` class:

```python
from vectorium import VectorDatabase
import numpy as np

# Create a new database named 'my_collection'
db = VectorDatabase('my_collection', dim=128)

# Add a new vector associated with the key 'my_key'
db.add('my_key', np.random.randn(128))

# Compare an input vector with the database
results = db.compare(np.random.randn(128), func='cosine', aggregate='mean')

# Remove a key from the database
db.remove('my_key')

# Save the database
db.save()

# Load the database
db.load('my_collection_path')
```

## Class: VectorDatabase

### Parameters
- `collection` - The name of the database. Will be used as a filename when saving/loading.
- `dim` (optional) - The dimensions of the vectors. If None, will be inferred from the first added vector.
- `collection_path` (optional) - The path where the database file will be stored.

### Methods

#### `add(key, vec)`
Add a new vector associated with the given key to the database. If the key already exists, the vector will be appended to the existing ones.

#### `remove(key)`
Removes the vectors associated with the given key from the database.

#### `compare(input_vector, func='cosine', aggregate='mean')`
Compares an input vector with the vectors in the database using the given compare function (default is 'cosine') and returns the results aggregated using the given aggregate function (default is 'mean').

#### `topk(input_vector, k=10, func='cosine', aggregate='mean', reverse=False)`
Similar to `compare`, but only returns the top `k` results.

#### `reset()`
Empties the database.

#### `save()`
Saves the database to a `.npz` file with the name of the collection.

#### `load(collection_path)`
Loads the database from a `.npz` file located at the given path.

#### `update()`
Updates the internal list representation of the database. This is called automatically after each `add`, `remove`, `reset`, `load` and `save` operation.

## Requirements

- Python 3.6+
- NumPy
- PyTorch

## Contribution

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
