import tensorflow as tf
import os
import utils
import style_transfer_tester
import argparse
import time

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--style_model', type=str, default='models/wave.ckpt', help='location for model file (*.ckpt)',
                        required=True)

    parser.add_argument('--content', type=str, default='content/female_knight.jpg',
                        help='File path of content image (notation in the paper : x)', required=True)

    parser.add_argument('--output', type=str, default='result.jpg',
                        help='File path of output image (notation in the paper : y_c)', required=True)

    parser.add_argument('--max_size', type=int, default=None, help='The maximum width or height of input images')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --style_model
    try:
        #Tensorflow r0.12 requires 3 files related to *.ckpt
        assert os.path.exists(args.style_model + '.index') and os.path.exists(args.style_model + '.meta') and os.path.exists(
            args.style_model + '.data-00000-of-00001')
    except:
        print('There is no %s'%args.style_model)
        print('Tensorflow r0.12 requires 3 files related to *.ckpt')
        print('If you want to restore any models generated from old tensorflow versions, this assert might be ignored')
        return None

    # --content
    try:
        assert os.path.exists(args.content)
    except:
        print('There is no %s' % args.content)
        return None

    # --max_size
    try:
        if args.max_size is not None:
            assert args.max_size > 0
    except:
        print('The maximum width or height of input image must be positive')
        return None

    # --output
    dirname = os.path.dirname(args.output)
    try:
        if len(dirname) > 0:
            os.stat(dirname)
    except:
        os.mkdir(dirname)

    return args

"""main"""
def main():

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # load content image
    content_image = utils.load_image(args.content, max_size=args.max_size)

    # open session
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True # to deal with large image
    sess = tf.Session(config=soft_config)

    # build the graph
    transformer = style_transfer_tester.StyleTransferTester(session=sess,
                                                            model_path=args.style_model,
                                                            content_image=content_image,
                                                            )
    # execute the graph
    start_time = time.time()
    output_image = transformer.test()
    end_time = time.time()

    # save result
    utils.save_image(output_image, args.output)

    # report execution time
    shape = content_image.shape #(batch, width, height, channel)
    print('Execution time for a %d x %d image : %f msec' % (shape[0], shape[1], 1000.*float(end_time - start_time)/60))

if __name__ == '__main__':
    main()
