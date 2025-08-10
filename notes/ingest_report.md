# Ingest Report (Step A)

- Source files: 64
- Patent records: 640
- Units kept (total): 31257
  - claims kept: 10198
  - desc kept:   21059
- Dropped (empty): 20819
- Dropped (too short < 15 chars): 101
- Dropped (duplicate within doc): 441

## Examples: empty after normalization
- [desc] ''
- [desc] ''
- [desc] ''
- [desc] ''
- [desc] ''

## Examples: too short (< 15)
- [desc] 'A-B-C (I)'
- [desc] 'A-B-C (I)'
- [desc] 'A-B-C (I)'
- [desc] 'DZ=DAb=1.1*DF'
- [claim] 'where “d'

## Schema
unit_id | doc_number | title | classification | section | idx | text | abstract | source_file
