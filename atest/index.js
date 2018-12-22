import * as tf from '@tensorflow/tfjs';

console.log('*************test');
const a = tf.scalar(2);
const b = tf.scalar(4);
const c = tf.scalar(8);

const result = predict(2);
result.print()

// 模型
function predict(input) {
    // y = a * x ^ 2 + b * x + c
    // More on tf.tidy in the next section
    return tf.tidy(() => {
        const x = tf.scalar(input);

        const ax2 = a.mul(x.square());
        const bx = b.mul(x);
        const y = ax2.add(bx).add(c);

        return y;
    });
}