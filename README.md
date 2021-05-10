# SPIN

## Maximal Frequent Subgraph Mining

Dear friends,\
This is my optimized implementation of SPIN.\
Paper: <https://doi.org/10.1145/1014052.1014123>

## How to use

### Install required packages

```
pip install -r requirements.txt
```

### Execute algorithm

```
python run.py output_file_name memory_log_file_name
              [-h] [-s MIN_SUPPORT] [-n NUM_GRAPHS]
              [-l LOWER_BOUND_OF_NUM_VERTICES]
              [-u UPPER_BOUND_OF_NUM_VERTICES] [-d DIRECTED] [-v VERBOSE]
              [-p PLOT] [-w WHERE] [-o OPTIMIZE]
              database_file_name
```

### Get results and memory usage

```
cat output_file_name
python trace.py memory_log_file_name
```
