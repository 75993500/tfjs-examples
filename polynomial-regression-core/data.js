import * as tf from '@tensorflow/tfjs';

/**
 * 生成随机数据
 * @param numPoints
 * @param coeff
 * @param sigma
 * @returns {{xs: Tensor<Rank>, ys: Tensor}}
 */
export function generateData(numPoints, coeff, sigma = 0.04) {
  return tf.tidy(() => {
    const [a, b, c, d] = [
      tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
      tf.scalar(coeff.d)
    ];

    // 随机均匀数据
    const xs = tf.randomUniform([numPoints], -1, 1);

    // Generate polynomial data
    //   生成多项式数据
    const three = tf.scalar(3, 'int32');
    const ys = a.mul(xs.pow(three))
      .add(b.mul(xs.square()))
      .add(c.mul(xs))
      .add(d)
      // Add random noise to the generated data
      // to make the problem a bit more interesting
      // 将随机噪声添加到生成的数据中使问题更有趣
      .add(tf.randomNormal([numPoints], 0, sigma));

    // Normalize the y values to the range 0 to 1.
    //   将y值标准化为0到1的范围。
    const ymin = ys.min();
    const ymax = ys.max();
    const yrange = ymax.sub(ymin);
    const ysNormalized = ys.sub(ymin).div(yrange);

    return {
      xs,
      ys: ysNormalized
    };
  })
}
/*
randomUniform 函数简介:
从均匀分布中返回随机值
random_uniform(
    shape,# 生成的张量的形状
    minval=0,
    maxval=None,
    dtype=tf.float32,
    seed=None,
    name=None)
返回值的范围默认是0到1的左闭右开区间，即[0，1)。
minval为指定最小边界，默认为1。
maxval为指定的最大边界，
如果是数据浮点型则默认为1，如果数据为整形则必须指定。
 */
