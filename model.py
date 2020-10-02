import tensorflow as tf
import config
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras import Model


class AwesomeNet(Model):
    def __init__(self):
        super(AwesomeNet, self).__init__()
        # 512x512x3
        self.conv1B1 = Conv2D(32, config.GRID_SIZE, 1, 'same', activation='relu')
        self.bn1B1 = BatchNormalization()
        self.conv2B1 = Conv2D(32, config.GRID_SIZE, 1, 'same', activation='relu')
        self.bn2B1 = BatchNormalization()
        self.conv3B1 = Conv2D(32, 1, 1, 'same', activation='relu')
        self.poolB1 = MaxPooling2D(2, 2)
        
        # 256x256x32
        self.conv1B2 = Conv2D(64, config.GRID_SIZE, 1, 'same', activation='relu')
        self.bn1B2 = BatchNormalization()
        self.conv2B2 = Conv2D(64, config.GRID_SIZE, 1, 'same', activation='relu')
        self.bn2B2 = BatchNormalization()
        self.conv3B2 = Conv2D(64, 1, 1, 'same', activation='relu')
        self.poolB2 = MaxPooling2D(2, 4)
        
        # 64x64x64
        self.conv1B3 = Conv2D(128, config.GRID_SIZE, 1, 'same', activation='relu')
        self.bn1B3 = BatchNormalization()
        self.conv2B3 = Conv2D(128, config.GRID_SIZE, 1, 'same', activation='relu')
        self.bn2B3 = BatchNormalization()
        self.conv3B3 = Conv2D(128, 1, 1, 'same', activation='relu')
        self.poolB3 = MaxPooling2D(2, 2)
        
        # 32x32x128
        self.conv1B4 = Conv2D(256, config.GRID_SIZE, 1, 'same', activation='relu')
        self.bn1B4 = BatchNormalization()
        self.conv2B4 = Conv2D(256, config.GRID_SIZE, 1, 'same', activation='relu')
        self.bn2B4 = BatchNormalization()
        self.conv3B4 = Conv2D(256, 2, 1, 'valid', activation='relu')
        self.poolB4 = MaxPooling2D(2, 2)
        
        # 8x8x256
        self.conv1B5 = Conv2D(256, config.GRID_SIZE, 1, activation='selu')
        self.bn1B5 = BatchNormalization()
        self.conv2B5 = Conv2D(256, 1, 1, 'same', activation='selu')
        self.bn2B5 = BatchNormalization()
        
        # 8x8x512
        self.convH1B1 = Conv2D(128, 1, 1, 'same', activation='selu')
        self.bn1H1 = BatchNormalization()
        self.do1H1 = Dropout(config.DROPOUT)
        # Detection Probabilities
        self.convH1B2d = Conv2D(1, 1, 1, 'same', activation='sigmoid')
        # Class Probabilities
        self.convH1B2 = Conv2D(256, 1, 1, 'same', activation='selu')
        self.convH1B3c = Conv2D(config.NUM_CLASSES, 1, 1, 'same', activation='softmax')
        
        # 8x8x512
        self.convH2B1 = Conv2D(128, 1, 1, 'same', activation='selu')
        self.bn1H2 = BatchNormalization()
        self.do1H2 = Dropout(config.DROPOUT)
        self.convH2B2c = Conv2D(2, 1, 1, 'same', activation='sigmoid')
        self.convH2B2b = Conv2D(2, 1, 1, 'same', activation='sigmoid')


    def call(self, x, training=False):
        # Forward pass of the 1st convolutional block
        x_a = self.conv1B1(x)
        x_a = self.bn1B1(x_a, training=training)
        x_b = self.conv2B1(x_a)
        x_b = self.bn2B1(x_b, training=training)
        x = tf.concat([x_a, x_b], -1)
        x = self.conv3B1(x)
        x = self.poolB1(x)

        # Forward pass of the 2nd convolutional block
        x_a = self.conv1B2(x)
        x_a = self.bn1B2(x_a, training=training)
        x_b = self.conv2B2(x_a)
        x_b = self.bn2B2(x_b, training=training)
        x = tf.concat([x_a, x_b], -1)
        x = self.conv3B2(x)
        x = self.poolB2(x)

        # Forward pass of the 3rd convolutional block
        x_a = self.conv1B3(x)
        x_a = self.bn1B3(x_a, training=training)
        x_b = self.conv2B3(x_a)
        x_b = self.bn2B3(x_b, training=training)
        x = tf.concat([x_a, x_b], -1)
        x = self.conv3B3(x)
        x = self.poolB3(x)

        # Forward pass of the 4th convolutional block
        x_a = self.conv1B4(x)
        x_a = self.bn1B4(x_a, training=training)
        x_b = self.conv2B4(x_a)
        x_b = self.bn2B4(x_b, training=training)
        x = tf.concat([x_a, x_b], -1)
        x = self.conv3B4(x)
        x = self.poolB4(x)

        # Forward pass of the 5th convolutional block
        x_a = self.conv1B5(x)
        x_a = self.bn1B5(x_a, training=training)
        x_b = self.conv2B5(x_a)
        x_b = self.bn2B5(x_b, training=training)
        x = tf.concat([x_a, x_b], -1)

        # Forward pass of the 1st head
        # This head is responsible for predicting whether an object exists in each cell
        # and the classes of the objects
        x_1 = self.convH1B1(x)
        x_h = self.bn1H1(x_1, training=training)
        x_h = self.do1H1(x_h, training=training)
        # Detection Probability
        x_d = self.convH1B2d(x_h)
        # Class Probability
        x_h = self.convH1B2(x)
        x_c = self.convH1B3c(x_h)


        # Forward pass of the 2nd head
        # This head is responsible for predicting the location of the centre of the objects in each cell
        # This head also predicts the distance to the verticies from the the predicted centre
        x_h = self.convH2B1(x)
        x_h = self.bn1H2(x_h, training=training)
        x_h = self.do1H2(x_h, training=training)
        # Bounding Box width/height
        x_bs = self.convH2B2b(x_h)
        # Bounding Box centre
        x_bc = self.convH2B2c(x_h)

        # The output of both heads are combined and returned
        return tf.concat([x_d, x_c, x_bc, x_bs], -1)