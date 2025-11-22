# Indexes Directory

This directory stores indexing files used for audio feature extraction and lookup.

## Structure
- `audio/`: Audio feature indexes
- `embeddings/`: Voice embedding indexes
- `features/`: Pre-computed feature indexes

## Usage
Indexes are created during training or preprocessing to speed up inference and enable efficient lookup of audio features.

## File Formats
- .index: Custom index files
- .faiss: FAISS vector indexes
- .npy: NumPy array indexes