## Methods

I simply created different methods to simulate some data and compare between these methods regarding their performance when increasing the sample size.

-   Method 1: unvectorized method using Python list;
-   Method 2: unvectorized method using Numpy array;
-   Method 3: partially vectorized method (i.e., this method still utilizes Python list and an explicit loop)
-   Method 4: fully vectorized method (i.e., only use Numpy array and vectorization provided by Numpy)

See the code below

```Python

def make_dummy_y_unvectorized1(x, vector_w, b, error_term):
    y = []
    m = x.shape[1]
    for i in range(m):
        y_i = 0
        for j in range(len(vector_w)):
            y_i += vector_w[j] * x[j, i]
        y_i = (y_i + b) * np.exp(error_term[i])

        y.append(y_i)
        y = np.array(y)
    return y

def make_dummy_y_unvectorized2(x, vector_w, b, error_term):
    m, n = x.shape
    y = np.zeros(n)
    for i in range(n):
        for j in range(m):
            y[i] += vector_w[j] * x[j, i]
    y = (y + b) * np.exp(error_term)
    return y


def make_dummy_y_vectorized1(x, vector_w, b, error_term):
    y = []
    for i in range(x.shape[1]):
        y.append((np.dot(vector_w, x[:, i]) + b) * np.exp(error_term[i]))
        y = np.array(y)
    return y


def make_dummy_y_vectorized2(x, vector_w, b, error_term):
    y = (np.dot(vector_w, x) + b) * np.exp(error_term)
    return y

```

## Results

<img src="/imgs/comparision.png" alt=""/>

In the comparison chart, method 1 and method 2 show a sharp increase in the time it takes to finish calculations as the amount of data grows, indicating they're not well-suited for large tasks. Method 3 improves on this, handling more data before slowing down. Method 4 - a fully vectorized method - stands out as the clear winner, maintaining a fast and consistent performance regardless of data size, showcasing its efficiency with heavy workloads.
