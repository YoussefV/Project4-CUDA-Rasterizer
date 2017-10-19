# **University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4:**

# **CUDA Rasterizer**





Tested on: Windows 10, Intel Core i7-7700HQ CPU @ 2.80 GHz, 8GB RAM, NVidia GeForce GTX 1050

 ![Built](https://img.shields.io/appveyor/ci/gruntjs/grunt.svg) ![Issues](https://img.shields.io/github/issues-raw/badges/shields/website.svg) ![CUDA 8.0](https://img.shields.io/badge/CUDA-8.0-green.svg?style=flat)  ![Platform](https://img.shields.io/badge/platform-Desktop-bcbcbc.svg)  ![Developer](https://img.shields.io/badge/Developer-Youssef%20Victor-0f97ff.svg?style=flat)




- [Features](#features)

- [In-Depth](#indepth)

- [Time Analysis](#time_analysis)

- [Bloopers](#bloopers)
 

 

____________________________________________________


 
The goal of this project was to run an algorithm that clears out all zeros from an array on the GPU using CUDA. This parallel reduction is done using the scan algorithm that computes the exclusive prefix sum. I also implemented a parallelized Radix Sort using the exclusive prefix sum algorithm developed.



### Things Done

#### Core Features

 - [x] Everything

 #### What Makes Me Special
 
  - [x] Perspective Correct Color Interpolation
  - [x] Instancing
  - [x] Super-Sampled Anti-Aliasing

![SSAA Cow Instanced](img/ssaa_instacow.gif)


### In-Depth:

#### Perspective Correct Color Interpolation

It is important to always correctly interpolate the color at each vertex, and so I did.

Here is what it looks like without perspective correct interpolation and with (side-by-side):

![Without Perspective-Correct Interpolation](img/not-persp-correct.PNG) ![With Perspective-Correct Interpolation](img/persp-correct.PNG)

Here they are in a nice GIF format that shows them back to back. (As with most color-dense GIFs, the colors are reduced for recording purposes, the colors in real life are exactly as they in the pictures above)

![Back-To-Back](img/persp-correction.gif)


#### Instancing

For instancing I added a preprocessor macro, `num_instances`, that defines the number of times that you want a mesh to be instanced. I then have a hard-coded array of transformation matrices that represent the transformations of each instance.

In the vertex shader, I loop over each instance and then transform each vertex `num_instances` times to correspond to the appropriate instance transformation.

I then go through the rest of the primitive rasterization as normal, with `num_instances * numPrimitives` primitives instead of the usual `numPrimitives`.

Here is what it looks like with the cow instanced 9 times:

![instaced_cow_9](img/instanced_cow.gif)

and here is what it looks like with it instanced 27 times!!

![instaced_cow_27](img/instanced_cow_27.gif)

#### Super-Sampled Anti-Aliasing:

For the super-sampled anti-aliasing (SSAA), multiply the fragment buffer by the preprocessor macro `SSAA_RES` which defines the scaling that each axis (`width`, `height`) is scaled by.

Here are the results, with FPS listed in the window title.

##### AA 1x1 (No anti-aliasing)

![anti-aliasing](img/aa_1.PNG)

##### AA 2x2

![anti-aliasing](img/aa_2.PNG)

##### AA 4x4

![anti-aliasing](img/aa_4.PNG)

##### AA 8x8

![anti-aliasing](img/aa_8.PNG)


### Time Analysis


### Bloopers

My best bloopers were mainly while creating the base render. I tried rendering using many debug views with varying levels of success. Here are some of my favorite.

##### "Neon Cow"

I was trying to rasterize the cow with normals showing, a depth buffer bug caused this beauty:

![neon_cow](img/neon_cow2.gif)

##### "The Cow Sees All"

I was trying to again modify the depth buffer, reversed the depth check and got this creepy situation where the cow follows you around if you move the camera in a certain angle:

![follow-cow](img/follow_cow.gif)

##### "Debug View: Eye Space Normal"

Here is a nice debug view of the cow with camera-space normals, the color segmentation is due to the GIF recording software reducing the color palette:

![irridescent](img/iridescent_cow.gif)
