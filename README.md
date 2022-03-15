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

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|n|uint|processing count|||
|x*|Array|input|n||
|y|Array|output|n||

### Vectorwise

```csharp
Vectorwise.___(uint n, uint stride, Array<T> v, Array<T> y);
Vectorwise.___(uint n, uint stride, Array<T> x, Array<T> v, Array<T> y);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|n|uint|processing count|||
|stride|uint|stride of x,y|||
|x|Array|input|(n, stride)||
|v|Array|input|stride||
|y|Array|output|(n, stride)|y &ne; v|

### Constant

```csharp
Constant.___(uint n, Array<T> x, T c, Array<T> y);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|n|uint|processing count|||
|x|Array|input|n||
|c|float or double|input value|||
|y|Array|output|n||

### Aggregate

```csharp
Aggregate.___(uint n, uint samples, uint stride, Array<T> x, Array<T> y);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|n|uint|processing count|||
|samples|uint|sampling count|||
|stride|uint|stride of x,y|||
|x|Array|input|(n, samples, stride)||
|y|Array|output|(n, stride)||

### Vectorise

```csharp
Vectorise.___(uint n, uint stride, Array<T> x, Array<T> y);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|n|uint|processing count|||
|stride|uint|stride of y|||
|x|Array|input|n||
|y|Array|output|(n, stride)|y &ne; x|

### Reorder

```csharp
Reorder.___(uint n, uint items, uint stride, Array<T> x, Array<T> y);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|n|uint|processing count|||
|items|uint|reordering count|||
|stride|uint|stride of x,y|||
|x|Array|input|(n, items, stride)||
|y|Array|output|(n, items, stride)|y &ne; x|

### Initialize

```csharp
Initialize.Zeroset(uint n, Array<T> y);
Initialize.Zeroset(uint index, uint n, Array<T> y);
Initialize.Clear(uint n, T c, Array<T> y);
Initialize.Clear(uint index, uint n, T c, Array<T> y);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|index|uint|processing offset, default=0|||
|n|uint|processing count|||
|c|float or double|filling value|||
|y|Array|output|index + n||

### Transform

```csharp
Transform.Transpose(uint n, uint r, uint s, uint stride, Array<T> x, Array<T> y);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|n|uint|batches|||
|r|uint|transposing axis length 1|||
|s|uint|transposing axis length 2|||
|stride|uint|stride of x,y|||
|x|Array|input|(n, r, s, stride)||
|y|Array|output|(n, s, r, stride)|y &ne; x|

### Affine
```csharp
Affine.Dotmul(uint na, uint nb, uint stride, Array<T> a, Array<T> b, Array<T> y);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|na|uint|processing count of a|||
|nb|uint|processing count of b|||
|stride|uint|stride of a,b|||
|a|Array|input|(na, stride)||
|b|Array|input|(nb, stride)||
|y|Array|output|(na, nb)|y &ne; a,b|

### Dense
```csharp
Dense.Forward(uint n, uint inch, uint outch, Array<T> x, Array<T> w, Array<T> y);
Dense.BackwardData(uint n, uint inch, uint outch, Array<T> dy, Array<T> w, Array<T> dx);
Dense.BackwardFilter(uint n, uint inch, uint outch, Array<T> x, Array<T> dy, Array<T> dw);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|n|uint|batches|||
|inch|uint|in channels|||
|outch|uint|out channels|||
|x/dx|Array|input/output|(n, inch)||
|y/dy|Array|output/input|(n, outch)||
|w/dw|Array|input/output|(outch, inch)||

### Convolution1D
```csharp
Convolution1D.Forward(uint n, uint ic, uint oc, uint iw, uint kw, 
                      PadMode padmode, 
                      Array<T> x, Array<T> w, Array<T> y);
Convolution1D.BackwardData(uint n, uint ic, uint oc, uint iw, uint kw, 
                      PadMode padmode, 
                      Array<T> dy, Array<T> w, Array<T> dx);
Convolution1D.BackwardFilter(uint n, uint ic, uint oc, uint iw, uint kw, 
                      PadMode padmode, 
                      Array<T> x, Array<T> dy, Array<T> dw);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|n|uint|batches|||
|ic|uint|in channels|||
|oc|uint|out channels|||
|iw|uint|in width|||
|kw|uint|kernel width||odd number|
|padmode|PadMode|||None,Zero,Edge|
|x/dx|Array|input/output|(n, iw, ic)||
|y/dy|Array|output/input|(n, ow, oc)|padmode=None: ow = iw-kw+1 <br/> padmode=Zero,Edge: ow = iw|
|w/dw|Array|input/output|(kw, oc, ic)|y &ne; x,w, dx &ne; dy,w, dw &ne; x,dy|

### Convolution2D
```csharp
Convolution2D.Forward(uint n, uint ic, uint oc, uint iw, uint ih, uint kw, uint kh, 
                      Array<T> x, Array<T> w, Array<T> y);
Convolution2D.BackwardData(uint n, uint ic, uint oc, uint iw, uint ih, uint kw, uint kh, 
                      Array<T> dy, Array<T> w, Array<T> dx);
Convolution2D.BackwardFilter(uint n, uint ic, uint oc, uint iw, uint ih, uint kw, uint kh, 
                      Array<T> x, Array<T> dy, Array<T> dw);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|n|uint|batches|||
|ic|uint|in channels|||
|oc|uint|out channels|||
|iw|uint|in width|||
|ih|uint|in height|||
|kw|uint|kernel width||odd number|
|kh|uint|kenrel height||odd number|
|x/dx|Array|input/output|(n, ih, iw, ic)||
|y/dy|Array|output/input|(n, oh, ow, oc)|oh = ih-kh+1, ow = iw-kw+1|
|w/dw|Array|input/output|(kh, kw, oc, ic)|y &ne; x,w, dx &ne; dy,w, dw &ne; x,dy|

### Convolution3D
```csharp
Convolution3D.Forward(uint n, uint ic, uint oc, uint iw, uint ih, uint id, uint kw, uint kh, uint kd, 
                      Array<T> x, Array<T> w, Array<T> y);
Convolution3D.BackwardData(uint n, uint ic, uint oc, uint iw, uint ih, uint id, uint kw, uint kh, uint kd, 
                      Array<T> dy, Array<T> w, Array<T> dx);
Convolution3D.BackwardFilter(uint n, uint ic, uint oc, uint iw, uint ih, uint id, uint kw, uint kh, uint kd, 
                      Array<T> x, Array<T> dy, Array<T> dw);
```

|parameter|type|note|shape|condition|
|---|---|---|---|---|
|T|float or double|type|||
|n|uint|batches|||
|ic|uint|in channels|||
|oc|uint|out channels|||
|iw|uint|in width|||
|ih|uint|in height|||
|id|uint|in depth|||
|kw|uint|kernel width||odd number|
|kh|uint|kenrel height||odd number|
|kd|uint|kenrel depth||odd number|
|x/dx|Array|input/output|(n, id, ih, iw, ic)||
|y/dy|Array|output/input|(n, od, oh, ow, oc)|od = id-kd+1, oh = ih-kh+1, <br/ >ow = iw-kw+1|
|w/dw|Array|input/output|(kd, kh, kw, oc, ic)|y &ne; x,w, dx &ne; dy,w,  <br/ >dw &ne; x,dy|

## Licence
[MIT](https://github.com/tk-yoshimura/AvxBlas/blob/main/LICENSE)

## Author

[T.Yoshimura](https://github.com/tk-yoshimura)
