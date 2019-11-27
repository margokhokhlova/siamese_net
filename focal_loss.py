import tensorflow as tf
from tensorflow.python.ops import array_ops

from keras import backend as K
import tensorflow as tf

def focal_loss_bin(y_true, y_pred, gamma=2, alpha=0.25):
    eps = 1e-12
    y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def focal_loss_margo(labels, y_pred, gamma=2, alpha=0.75):
    eps = 1e-12
    y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
    labels = tf.to_float(labels)  # int -> float
    return K.sum(-labels * alpha * ((1 - y_pred) ** gamma) * tf.log(y_pred)
           - (1 - labels) * (1 - alpha) * (y_pred ** gamma) * tf.log(1 - y_pred))


def focal_loss_sigmoid_on_2_classification(labels, logits, alpha=0.5, gamma=2):
    """
    description:
        基于logtis输出的2分类focal loss计算
    计算公式：
        pt = p if label=1, else pt = 1-p； p表示判定为类别1（正样本）的概率
        focal loss = - alpha * (1-pt) ** (gamma) * log(pt)

    Args:
        labels: [batch_size], dtype=int32，值为0或者1
        logits: [batch_size], dtype=float32，输入为logits值
        alpha: 控制样本数目的权重，当正样本数目大于负样本时，alpha<0.5，反之，alpha>0.5。
        gamma：focal loss的超参数
    Returns:
        tensor: [batch_size]
    """
    y_pred = tf.nn.sigmoid(logits)  # 转换成概率值
    labels = tf.to_float(labels)  # int -> float

    """
    if label=1, loss = -alpha * ((1 - y_pred) ** gamma) * tf.log(y_pred)
    if label=0, loss = - (1 - alpha) * (y_pred ** gamma) * tf.log(1 - y_pred)
    alpha=0.5，表示赋予不考虑数目差异，此时权重是一致的
    将上面两个标签整合起来，得到下面统一的公式：
        focal loss = -alpha * (1-p)^gamma * log(p) - (1-apha) * p^gamma * log(1-p)
    """
    loss = -labels * alpha * ((1 - y_pred) ** gamma) * tf.log(y_pred) \
           - (1 - labels) * (1 - alpha) * (y_pred ** gamma) * tf.log(1 - y_pred)
    return loss



if __name__ == '__main__':
    tf.InteractiveSession()  # run an interactive session in Tf.
    y_true = tf.ones([2, 3], dtype="float32")
    y_pred = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    loss =focal_loss_bin(y_true, y_pred)
    print(loss.eval())
    y_pred = tf.constant([[1.0,1.0, 1.0],[0.0,0.0, 0.0]])
    loss = focal_loss_bin (y_true, y_pred)
    print(loss.eval())

