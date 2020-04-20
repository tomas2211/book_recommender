# Simple book recommender

Project discovering creative approaches to recommender systems.

See presentation for evaluation and practical info: [simple_book_recommender.pdf](simple_book_recommender.pdf).

Using [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).

## Setup

1. Clone the repo `git@github.com:tomas2211/ds_books.git`
2. Install requrements `pip install -r requirements.txt`
3. Download and unzip the dataset & trained models `./download_data.sh`
4. (Not required - only when training node2vec model `git submodule init && git submodule update`)

## Demo

For a quick demo, run:
```shell script
python eval_qualitative.py --interactive --format
```

The four tested models will be loaded, and you will be able to enter queries (book names).


## API
The kNN model is reachable through a simple API (change book-name to query):

[https://abiding-ripple-272918.ew.r.appspot.com/query?name=book-name](https://abiding-ripple-272918.ew.r.appspot.com/query?name=book-name)

Pass 'format' parameter for a human-readable response:

[https://abiding-ripple-272918.ew.r.appspot.com/query?format=1&name=book-name](https://abiding-ripple-272918.ew.r.appspot.com/query?name=book-name)