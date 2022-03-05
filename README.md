# AvxBlas
 Avx Basic Linear Algebra Subroutine and Neural Network Kernel Library

# *under development !*

## Requirement
.NET 5.0  
AVX2 suppoted CPU. (Intel:Haswell(2013)-, AMD:Excavator(2015)-)

## Install

[Download DLL](https://github.com/tk-yoshimura/AvxBlas/releases)  
[Download Nuget](https://www.nuget.org/packages/tyoshimura.avxblas.ode/)  

- To install, just import the DLL.
- This library does not change the environment at all.

## Usage

```csharp
// make input array
Array<float> x1 = new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
Array<float> x2 = new float[] { 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5 };

// make zeroset output array
Array<float> y = new(x1.Length);

// orerate
Elementwise.Add(9, x1, x2, y);

// check
float[] t = new float[] { 6, 8, 10, 12, 14, 5, 7, 9, 11, 0, 0 };
CollectionAssert.AreEqual(t, (float[])y);
```

## Operations

### Elementwise

```csharp
Elementwise.###(uint n, Array<T> x1, Array<T> x2, Array<T> y);
```

|parameter|type|note|
|---|---|---|
|T|float or double|type|
|n|uint|count of processing|
|x1|Array|input, length &geq; n|
|x2|Array|input, length &geq; n|
|y|Array|output, length &geq; n|

### Vectorwise

```csharp
Vectorwise.###(uint n, uint incx, Array<T> x, Array<T> v, Array<T> y);
```

|parameter|type|note|
|---|---|---|
|T|float or double|type|
|n|uint|count of processing|
|incx|uint|stride of x,y|
|x|Array|input, length &geq; n * incx|
|v|Array|input, length &geq; incx|
|y|Array|output, length &geq; n * incx|

## Licence
[MIT](https://github.com/tk-yoshimura/AvxBlas/blob/main/LICENSE)

## Author

[T.Yoshimura](https://github.com/tk-yoshimura)
