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

// operate
Elementwise.Add(9, x1, x2, y);

// check
float[] t = new float[] { 6, 8, 10, 12, 14, 5, 7, 9, 11, 0, 0 };
CollectionAssert.AreEqual(t, (float[])y);
```

## Operations

### Elementwise

```csharp
Elementwise.___(uint n, Array<T> x, Array<T> y);
Elementwise.___(uint n, Array<T> x1, Array<T> x2, Array<T> y);
Elementwise.___(uint n, Array<T> x1, Array<T> x2, Array<T> x3, Array<T> y);
```

|parameter|type|note|condition|
|---|---|---|---|
|T|float or double|type||
|n|uint|processing count||
|x*|Array|input|length &geq; n|
|y|Array|output|length &geq; n|

### Vectorwise

```csharp
Vectorwise.___(uint n, uint stride, Array<T> v, Array<T> y);
Vectorwise.___(uint n, uint stride, Array<T> x, Array<T> v, Array<T> y);
```

|parameter|type|note|condition|
|---|---|---|---|
|T|float or double|type||
|n|uint|processing count||
|stride|uint|stride of x,y||
|x|Array|input|length &geq; n x stride|
|v|Array|input|length &geq; stride|
|y|Array|output|length &geq; n x stride, y &ne; v|

### Constant

```csharp
Constant.___(uint n, Array<T> x, T c, Array<T> y);
```

|parameter|type|note|condition|
|---|---|---|---|
|T|float or double|type||
|n|uint|processing count||
|x|Array|input|length &geq; n|
|c|float or double|input value||
|y|Array|output|length &geq; n|

### Aggregation

```csharp
Aggregation.___(uint n, uint samples, uint stride, Array<T> x, Array<T> y);
```

|parameter|type|note|condition|
|---|---|---|---|
|T|float or double|type||
|n|uint|processing count||
|samples|uint|sampling count||
|stride|uint|stride of x,y||
|x|Array|input|length &geq; n x samples x stride|
|y|Array|output|length &geq; n x stride, y &ne; x|

### Vectorise

```csharp
Vectorise.___(uint n, uint stride, Array<T> x, Array<T> y);
```

|parameter|type|note|condition|
|---|---|---|---|
|T|float or double|type||
|n|uint|processing count||
|samples|uint|sampling count||
|stride|uint|stride of x,y||
|x|Array|input|length &geq; n|
|y|Array|output|length &geq; n x stride, y &ne; x|

### Reorder

```csharp
Reorder.___(uint n, uint items, uint stride, Array<T> x, Array<T> y);
```

|parameter|type|note|condition|
|---|---|---|---|
|T|float or double|type||
|n|uint|processing count||
|items|uint|reordering count||
|stride|uint|stride of x,y||
|x|Array|input|length &geq; n x samples x stride|
|y|Array|output|length &geq; n x samples x stride, y &ne; x|

### Initialize

```csharp
Initialize.Zeroset(uint n, Array<T> y);
Initialize.Zeroset(uint index, uint n, Array<T> y);
Initialize.Clear(uint n, T c, Array<T> y);
Initialize.Clear(uint index, uint n, T c, Array<T> y);
```

|parameter|type|note|condition|
|---|---|---|---|
|T|float or double|type||
|index|uint|processing offset, default=0||
|n|uint|processing count||
|c|float or double|filling value||
|y|Array|output|length &geq; index + n|

## Licence
[MIT](https://github.com/tk-yoshimura/AvxBlas/blob/main/LICENSE)

## Author

[T.Yoshimura](https://github.com/tk-yoshimura)
