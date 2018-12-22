import * as tf from '@tensorflow/tfjs';

// This is a helper class for loading and managing MNIST data specifically.
// It is a useful example of how you could create your own data manager class
// for arbitrary data though. It's worth a look :)
// 这是一个专门用于加载和管理MNIST数据的辅助类。
// 它是如何创建自己的数据管理器类的有用示例但是对于任意数据。 值得一看:)
import {IMAGE_H, IMAGE_W, MnistData} from './data';

// This is a helper class for drawing loss graphs and MNIST images to the
// window. For the purposes of understanding the machine learning bits, you can
// largely ignore it
// 这是一个帮助类，用于绘制损耗图和MNIST图像窗口。 为了理解机器学习位，您可以很大程度上忽略它
import * as ui from './ui';

/**
 * Creates a convolutional neural network (Convnet) for the MNIST data.
 * 为MNIST数据创建卷积神经网络（Convnet）。
 * @returns {tf.Model} An instance of tf.Model.
 */
function createConvModel() {
  // Create a sequential neural network model. tf.sequential provides an API
  // for creating "stacked" models where the output from one layer is used as
  // the input to the next layer.
  //   创建顺序神经网络模型。 tf.sequential提供API
  //   用于创建“堆叠”模型，其中使用来自一个层的输出下一层的输入。
  const model = tf.sequential();

  // The first layer of the convolutional neural network plays a dual role:
  // it is both the input layer of the neural network and a layer that performs
  // the first convolution operation on the input. It receives the 28x28 pixels
  // black and white images. This input layer uses 16 filters with a kernel size
  // of 5 pixels each. It uses a simple RELU activation function which pretty
  // much just looks like this: __/
//     卷积神经网络的第一层起着双重作用：
//    它既是神经网络的输入层，也是执行的层对输入的第一个卷积运算。 它接收28x28像素黑白图像。
//     此输入层使用16个内核大小的过滤器每个5像素。
//     它使用简单的RELU激活功能很多看起来像这样：__ /
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_H, IMAGE_W, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));

  // After the first layer we include a MaxPooling layer. This acts as a sort of
  // downsampling using max values in a region instead of averaging.
  //   在第一层之后，我们包含一个MaxPooling图层。 这就像一种使用区域中的最大值而不是平均值进行下采样。
  // https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

  // Our third layer is another convolution, this time with 32 filters.
  //   我们的第三层是另一个卷积，这次是32个过滤器。
  model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));

  // Max pooling again.
  //   最大限度地汇集。
  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

  // Add another conv2d layer.
  //   添加另一个conv2d图层。
  model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  //   现在我们将2D滤镜的输出展平为1D矢量以进行准备它输入到我们的最后一层。
  //   喂食时这是常见的做法更高维数据到最终分类输出层。
  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({units: 64, activation: 'relu'}));

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
  // represent numbers, but it's the same idea if you had classes that
  // represented other entities like dogs and cats (two output classes: 0, 1).
  // We use the softmax function as the activation for the output layer as it
  // creates a probability distribution over our 10 classes so their output
  // values sum to 1.
//     我们的最后一层是一个密集层，有10个输出单元，每个单元一个输出类（即0,1,2,3,4,5,6,7,8,9）。
//     这里的课程实际上代表数字，但如果你有类，那就是同样的想法代表其他实体，如狗和猫（两个输出类：0,1）。
//    我们使用softmax函数作为输出层的激活在我们的10个类中创建概率分布，以便输出值总和为1。
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  return model;
}

/**
 * Creates a model consisting of only flatten, dense and dropout layers.
 * 创建一个仅包含展平，密集和辍学图层的模型。
 *
 * The model create here has approximately the same number of parameters
 * (~31k) as the convnet created by `createConvModel()`, but is
 * expected to show a significantly worse accuracy after training, due to the
 * fact that it doesn't utilize the spatial information as the convnet does.
 * 此处创建的模型具有大致相同数量的参数（~31k）作为由`createConvModel（）`创建的convnet，
 * 但是由于这个原因，预计在训练后会显示出更差的准确性
 * 事实上，它不像投票机那样利用空间信息。
 *
 * This is for comparison with the convolutional network above.
 * 这是为了与上面的卷积网络进行比较。
 *
 * @returns {tf.Model} An instance of tf.Model.
 */
function createDenseModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape: [IMAGE_H, IMAGE_W, 1]}));
  model.add(tf.layers.dense({units: 42, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  return model;
}

/**
 * Compile and train the given model.
 * 编译并训练给定的模型。
 *
 * @param {*} model The model to
 */
async function train(model, onIteration) {
  ui.logStatus('Training model...');

  // Now that we've defined our model, we will define our optimizer. The
  // optimizer will be used to optimize our model's weight values during
  // training so that we can decrease our training loss and increase our
  // classification accuracy.
  //   现在我们已经定义了我们的模型，我们将定义我们的优化器。该优化器将用于优化模型的重量值
  //   培训，以便我们可以减少我们的培训损失，增加我们的分类准确性。

  // The learning rate defines the magnitude by which we update our weights each
  // training step. The higher the value, the faster our loss values converge,
  // but also the more likely we are to overshoot optimal parameters
  // when making an update. A learning rate that is too low will take too long
  // to find optimal (or good enough) weight parameters while a learning rate
  // that is too high may overshoot optimal parameters. Learning rate is one of
  // the most important hyperparameters to set correctly. Finding the right
  // value takes practice and is often best found empirically by trying many
  // values.
//     学习率定义了我们每次更新权重的幅度训练步骤。 值越高，我们的损失值收敛得越快，
//    但我们越有可能超越最佳参数在进行更新时。 学习率太低会花费太长时间
//     在学习率的同时找到最佳（或足够好）的权重参数
//     太高可能会超出最佳参数。 学习率是其中之一要正确设置的最重要的超参数。
//     寻找合适的人选价值需要实践，并且通常通过尝试很多来经验最好地找到值。
  const LEARNING_RATE = 0.01;

  // We are using rmsprop as our optimizer.
  // An optimizer is an iterative method for minimizing an loss function.
  // It tries to find the minimum of our loss function with respect to the
  // model's weight parameters.
//     我们使用rmsprop作为我们的优化器。
//    优化器是用于最小化损失函数的迭代方法。
//    它试图找到我们的损失函数的最小值模型的权重参数。
  const optimizer = 'rmsprop';

  // We compile our model by specifying an optimizer, a loss function, and a
  // list of metrics that we will use for model evaluation. Here we're using a
  // categorical crossentropy loss, the standard choice for a multi-class
  // classification problem like MNIST digits.
  // The categorical crossentropy loss is differentiable and hence makes
  // model training possible. But it is not amenable to easy interpretation
  // by a human. This is why we include a "metric", namely accuracy, which is
  // simply a measure of how many of the examples are classified correctly.
  // This metric is not differentiable and hence cannot be used as the loss
  // function of the model.
//     我们通过指定优化器，损失函数和a来编译模型
//     我们将用于模型评估的指标列表。 我们在这里使用分类的交叉熵损失，多类的标准选择
//     像MNIST数字这样的分类问题。
//    分类的交叉熵损失是可微分的，因此产生模型训练可行。 但它不易于解释由人类。
//     这就是为什么我们包含一个“度量”，即准确度，即只是衡量有多少例子被正确分类的衡量标准。
//     该指标不可区分，因此不能用作损失模型的功能。
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Batch size is another important hyperparameter. It defines the number of
  // examples we group together, or batch, between updates to the model's
  // weights during training. A value that is too low will update weights using
  // too few examples and will not generalize well. Larger batch sizes require
  // more memory resources and aren't guaranteed to perform better.
  //   批量大小是另一个重要的超参数。 它定义了数量我们将模型更新组合在一起或批处理的示例
  //   训练期间的重量。 值太低会使用更新权重示例太少，不会很好地概括。
  //   需要更大的批量更多的内存资源，并不能保证更好的性能。
  const batchSize = 320;

  // Leave out the last 15% of the training data for validation, to monitor
  // overfitting during training.
  //   遗漏最后15％的培训数据进行验证，进行监控在训练期间过度拟合。
  const validationSplit = 0.15;

  // Get number of training epochs from the UI.
  //   从UI获取训练时期的数量。
  const trainEpochs = ui.getTrainEpochs();

  // We'll keep a buffer of loss and accuracy values over time.
  //   随着时间的推移，我们将保持损失和准确值的缓冲。
  let trainBatchCount = 0;

  const trainData = data.getTrainData();
  const testData = data.getTestData();

  const totalNumBatches =
      Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
      trainEpochs;

  // During the long-running fit() call for model training, we include
  // callbacks, so that we can plot the loss and accuracy values in the page
  // as the training progresses.
  //   在长期运行fit（）调用模型培训期间，我们包括回调，以便我们可以在页面中绘制损失和准确度值随着训练的进行。
  let valAcc;
  await model.fit(trainData.xs, trainData.labels, {
    batchSize,
    validationSplit,
    epochs: trainEpochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainBatchCount++;
        ui.logStatus(
            `Training... (` +
            `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
            ` complete). To stop training, refresh or close page.`);
        ui.plotLoss(trainBatchCount, logs.loss, 'train');
        ui.plotAccuracy(trainBatchCount, logs.acc, 'train');
        if (onIteration && batch % 10 === 0) {
          onIteration('onBatchEnd', batch, logs);
        }
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        valAcc = logs.val_acc;
        ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
        ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
        if (onIteration) {
          onIteration('onEpochEnd', epoch, logs);
        }
        await tf.nextFrame();
      }
    }
  });

  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
  ui.logStatus(
      `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
      `Final test accuracy: ${testAccPercent.toFixed(1)}%`);
}

/**
 * Show predictions on a number of test examples.
 * 显示许多测试示例的预测。
 * @param {tf.Model} model The model to be used for making the predictions.
 */
async function showPredictions(model) {
  const testExamples = 100;
  const examples = data.getTestData(testExamples);

  // Code wrapped in a tf.tidy() function callback will have their tensors freed
  // from GPU memory after execution without having to call dispose().
  // The tf.tidy callback runs synchronously.
//     包含在tf.tidy（）函数回调中的代码将释放其张量执行后从GPU内存中无需调用dispose（）。
//     tf.tidy回调同步运行。
  tf.tidy(() => {
    const output = model.predict(examples.xs);

    // tf.argMax() returns the indices of the maximum values in the tensor along
    // a specific axis. Categorical classification tasks like this one often
    // represent classes as one-hot vectors. One-hot vectors are 1D vectors with
    // one element for each output class. All values in the vector are 0
    // except for one, which has a value of 1 (e.g. [0, 0, 0, 1, 0]). The
    // output from model.predict() will be a probability distribution, so we use
    // argMax to get the index of the vector element that has the highest
    // probability. This is our prediction.
    // (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
    // dataSync() synchronously downloads the tf.tensor values from the GPU so
    // that we can use them in our normal CPU JavaScript code
    // (for a non-blocking version of this function, use data()).
    //   tf.argMax（）返回张量中最大值的索引特定的轴。
    //   经常这样的分类分类任务将类表示为单热矢量。
    //   单热矢量是一维矢量每个输出类的一个元素。
    //   向量中的所有值均为0除了一个，其值为1（例如[0,0,0,1,0]）。
    //   该model.predict（）的输出将是一个概率分布，所以我们使用argMax获取具有最高的向量元素的索引可能性。
    //   这是我们的预测。（例如argmax（[0.07,0.1,0.03,0.75,0.05]）== 3）dataSync（）同步从GPU下载tf.tensor值
    //   我们可以在普通的CPU JavaScript代码中使用它们
    //   （对于此函数的非阻塞版本，请使用data（））。
    const axis = 1;
    const labels = Array.from(examples.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

    ui.showTestResults(examples, predictions, labels);
  });
}

function createModel() {
  let model;
  const modelType = ui.getModelTypeId();
  if (modelType === 'ConvNet') {
    model = createConvModel();
  } else if (modelType === 'DenseNet') {
    model = createDenseModel();
  } else {
    throw new Error(`Invalid model type: ${modelType}`);
  }
  return model;
}

let data;
async function load() {
  data = new MnistData();
  await data.load();
}

// This is our main function. It loads the MNIST data, trains the model, and
// then shows what the model predicted on unseen test data.
// 这是我们的主要功能。 它加载MNIST数据，训练模型，和
// 然后显示模型预测的看不见的测试数据。
ui.setTrainButtonCallback(async () => {
  ui.logStatus('Loading MNIST data...');
  await load();

  ui.logStatus('Creating model...');
  const model = createModel();
  model.summary();

  ui.logStatus('Starting model training...');
  await train(model, () => showPredictions(model));
});
