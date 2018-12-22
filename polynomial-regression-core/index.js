import * as tf from '@tensorflow/tfjs';
import {generateData} from './data';
import {plotData, plotDataAndPredictions, renderCoefficients} from './ui';

/**
 * https://www.w3cschool.cn/tensorflowjs/tensorflowjs-y6kx2q2c.html
 * https://js.tensorflow.org/tutorials/fit-curve.html
 *
 * We want to learn the coefficients that give correct solutions to the
 * following cubic equation:
 * 我们想要学习能够为其提供正确解决方案的系数以下方程式：
 *      y = a * x^3 + b * x^2 + c * x + d
 * In other words we want to learn values for:
 *  换句话说，我们想要训练学习:
 *      a
 *      b
 *      c
 *      d
 * Such that this function produces 'desired outputs' for y when provided
 * with x. We will provide some examples of 'xs' and 'ys' to allow this model
 * to learn what we mean by desired outputs and then use it to produce new
 * values of y that fit the curve implied by our example.
 * 这样，当提供时，该函数为y产生“期望输出”用x。
 * 我们将提供一些'xs'和'ys'的例子来允许这个模型通过期望的输出来学习我们的意思，
 * 然后用它来产生新的符合我们示例所暗示的曲线的y值。
 */

// Step 1. Set up variables, these are the things we want the model
// to learn in order to do prediction accurately. We will initialize
// them with random values.
// 设置变量，这些是我们想要的模型学习以便准确地进行预测。 我们将初始化它们具有随机值。
// 第1步：设置变量
// 首先，我们需要创建一些变量。即开始我们是不知道a、b、c、d的值的，所以先给他们一个随机数，入戏所示：
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));


// Step 2. Create an optimizer, we will use this later. You can play
// with some of these values to see how the model performs.
// 创建一个优化器，稍后我们将使用它。 你可以玩使用其中一些值来查看模型的执行情况。
// SGD （Stochastic Gradient Descent）优化器，即随机梯度下降。SGD的工作原理就是利用数据中任意的点的梯度
// 以及使用它们的值来决定增加或者减少我们模型中系数的值。
const numIterations = 75;
// 学习率（learning rate）会控制模型调整幅度将会有多大。低的学习率会使得学习过程运行的更慢一些
// （更多的训练迭代获得更符合数据的系数），而高的学习率将会加速学习过程但是将会导致最终的模型可能在正确值周围摇摆。
// 简单的说，你既想要学的快，又想要学的好，这是不可能的。
const learningRate = 0.5; // 学习率为0.5的SGD优化器
const optimizer = tf.train.sgd(learningRate); // SGD 优化器

// Step 3. Write our training process functions.
// 写下我们的培训流程功能。

/*
 *  预测函数: 构建模型
 * This function represents our 'model'. Given an input 'x' it will try and
 * predict the appropriate output 'y'.
 * 此功能代表我们的“模型”。 给定输入'x'将尝试和预测适当的输出'y'。
 *
 * It is also sometimes referred to as the 'forward' step of our training
 * process. Though we will use the same function for predictions later.
 * 它有时也被称为我们培训的“前进”步骤处理。 虽然我们稍后会使用相同的功能进行预测。
 *
 * @return number predicted y value  数字预测y值
 */
function predict(x) {
    // y = a * x ^ 3 + b * x ^ 2 + c * x + d
  return tf.tidy(() => { // tidy函数进行内存管理,其作用:为执行一个函数并清除所有创建的中间张量
    return a.mul(x.pow(tf.scalar(3, 'int32'))) // a * x^3
      .add(b.mul(x.square())) // + b * x ^ 2
      .add(c.mul(x)) // + c * x
      .add(d); // + d
  });
}

/*
定义损失函数
 * This will tell us how good the 'prediction' is given what we actually
 * expected.
 * 这将告诉我们实际上给出的“预测”有多好预期。
 *
 * prediction is a tensor with our predicted y values.
 * labels is a tensor with the y values the model should have predicted.
 * 预测是我们预测的y值的张量。标签是一个张量，具有模型应该预测的y值。
 * 此处损失函数使用 MSE（均方误差，mean squared error）作为我们的损失函数
 * MSE的计算非常简单，就是先根据给定的x得到实际的y值与预测得到的y值之差 的平方，然后在对这些差的平方求平均数即可。
 * prediction: 预测值
 * labels: 实际值
 */
function loss(prediction, labels) {
  // Having a good error function is key for training a machine learning model
  //   具有良好的错误功能是培训机器学习模型的关键
  // 将labels（实际的值）进行抽象,然后获取平均数.
  // square() 平方函数
  // mean() 平均值函数
  const error = prediction.sub(labels).square().mean();
  return error;
}

/*
 * 训练迭代器: 它会不断地运行SGD优化器来使不断修正、完善模型的系数来减小损失（MSE）
 * This will iteratively train our model.
 * 这将迭代地训练我们的模型。
 *
 * xs - training data x values x值
 * ys — training data y values y值
 * numIterations 制定的迭代次数
 */
async function train(xs, ys, numIterations) {
  for (let iter = 0; iter < numIterations; iter++) { // 进行 numIterations 此迭代
    // optimizer.minimize is where the training happens.
    //   optimizer.minimize是培训发生的地方。

    // The function it takes must return a numerical estimate (i.e. loss)
    // of how well we are doing using the current state of
    // the variables we created at the start.
    //   它所需的功能必须返回一个数值估计（即损失）我们使用当前状态的表现如何我们在开始时创建的变量。

    // This optimizer does the 'backward' step of our training process
    // updating variables defined previously in order to minimize the
    // loss.
    //   此优化程序执行我们培训过程的“后退”步骤更新先前定义的变量以便最小化失利。
    optimizer.minimize(() => { // 每次迭代调用optimizer优化器的minimize函数
      // Feed the examples into the model
      //   将示例提供给模型
      const pred = predict(xs); // 进行预测
      return loss(pred, ys); // 返回损失值
    });

    // Use tf.nextFrame to not block the browser.
    //   使用tf.nextFrame不阻止浏览器。
    await tf.nextFrame();
  }
}

// 学习系数
async function learnCoefficients() {
  const trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5}; // 原始系数
  const trainingData = generateData(100, trueCoefficients);

  // Plot original data
  //   绘制原始数据
  renderCoefficients('#data .coeff', trueCoefficients);
  await plotData('#data .plot', trainingData.xs, trainingData.ys)

  // See what the predictions look like with random coefficients
  //   查看随机系数的预测结果
  renderCoefficients('#random .coeff', {
    a: a.dataSync()[0],
    b: b.dataSync()[0],
    c: c.dataSync()[0],
    d: d.dataSync()[0],
  });
  // 预测前
  const predictionsBefore = predict(trainingData.xs); // 进行预测
  // 并预测
  await plotDataAndPredictions(
      '#random .plot', trainingData.xs, trainingData.ys, predictionsBefore);

  // Train the model!
  //   训练模型, 循环迭代
  await train(trainingData.xs, trainingData.ys, numIterations);

  // See what the final results predictions are after training.
  //   了解培训后的最终结果预测。
  renderCoefficients('#trained .coeff', {
    a: a.dataSync()[0],
    b: b.dataSync()[0],
    c: c.dataSync()[0],
    d: d.dataSync()[0],
  });
  // 预测后
  const predictionsAfter = predict(trainingData.xs); // 进行预测
  await plotDataAndPredictions(
      '#trained .plot', trainingData.xs, trainingData.ys, predictionsAfter);

  predictionsBefore.dispose();
  predictionsAfter.dispose();
}


learnCoefficients();
