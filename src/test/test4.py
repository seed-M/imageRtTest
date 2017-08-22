import argparse
import sys
import tensorflow as tf

def main(_):
    print('arg1={0}'.format(FLAGS.arg1))
    print('arg2={0}'.format(FLAGS.arg2))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'arg1',
      type=int,
      default=5,
      help='arg1 mandatory'
  )
  parser.add_argument(
      '--arg2',
      type=str,
      default='this is arg2',
      help='optional arg2'
  )
FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)