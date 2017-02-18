# Fast Style Transfer

A tensorflow implementation of fast style transfer described in the papers:
* [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/) by Johnson
* [Instance Normalization](https://arxiv.org/abs/1607.08022) by Ulyanov

The implementation is coincided with the paper both in variable-names and algorithms so that a reader of the paper can understand the code without too much effort.

## Usage

### Prerequisites
1. Tensorflow
2. Python packages : numpy, scipy, PIL(or Pillow), matplotlib
3. Pretrained VGG19 file : [imagenet-vgg-verydeep-19.mat](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Please download the file from link above.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Save the file under `pre_trained_model`

4. MSCOCO train2014 DB : [LINK](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Please download the file from link above.  (Notice that the file size is over 12GB!!)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Extract images to `train2014`.

## Sample results

### Female Knight
The source image is from https://www.artstation.com/artwork/4zXxW

Results were obtained from default setting except `--max_size 1920`.  
An image was rendered approximately after 300ms on  GTX 980 ti.

<p align='left'>
<img src = 'content/female_knight.jpg' height="200px">
<br>
<img src = 'samples/female_knight_wave.jpg' height = '200px'>
<img src = 'style/thumbs/wave.jpg' height = '200px'>
<img src = 'samples/female_knight_the_scream.jpg' height = '200px'>
<img src = 'style/thumbs/the_scream.jpg' height = '200px'>
<br>
<img src = 'samples/female_knight_la_muse.jpg' height = '200px'>
<img src = 'style/thumbs/la_muse.jpg' height = '200px'>
<img src = 'samples/female_knight_rain_princess.jpg' height = '200px'>
<img src = 'style/thumbs/rain_princess.jpg' height = '200px'>
<br>
<img src = 'samples/female_knight_shipwreck.jpg' height = '200px'>
<img src = 'style/thumbs/the_shipwreck_of_the_minotaur.jpg' height = '200px'>
<img src = 'samples/female_knight_udnie.jpg' height = '200px'>
<img src = 'style/thumbs/udnie.jpg' height = '200px'>
<br>
</p>

## Acknowledgements
This implementation has been tested with Tensorflow r0.12 on Windows 10 and Ubuntu 14.04.
