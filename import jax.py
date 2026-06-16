import jax.numpy as jnp
import flax.nnx 


class ResnetBlock(nnx.Module):

    def __init__(self, in_features: int, out_features: int, stride: int = 1, * , rngs: nnx.Rngs):

        self.conv1 = nnx.Conv(in_features, out_features, kernel_size=(3, 3), 
                            strides=stride, padding='SAME', use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_features, rngs=rngs)

        self.conv2 = nnx.Conv(out_features, out_features, kernel_size=(3, 3), 
                            strides=1, padding='SAME', use_bias=False, rngs=rngs)        
        self.bn2 = nnx.BatchNorm(out_features, rngs=rngs)

        if in_features != out_features or stride != 1:
            self.shortcut_conv = nnx.Conv(in_features, out_features, kernel_size=(1, 1), strides=stride, padding='SAME', use_bias=False, rngs=rngs)
            self.shortcut_bn = nnx.BatchNorm(out_features, rngs=rngs)
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None       
        
    def __call__(self, x, deterministic: bool):
        residual = x
    
        #
        y = self.conv1(x)
        y = self.bn1(y, use_running_average=deterministic)
        y = nnx.relu(y)
        
        y = self.conv2(y)
        y = self.bn2(y, use_running_average=deterministic)
        
        # 2. shortcut (resnet)
        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(residual)
            residual = self.shortcut_bn(residual, use_running_average=deterministic)
            
        # 3. rejoin paths
        return nnx.relu(residual + y)

class LearnabilityResNet(nnx.Module):
    """A Small ResNet customized for grid mapping, written in flax.nnx."""
    
    def __init__(self, *, rngs: nnx.Rngs):
        # stem
        self.stem_conv = nnx.Conv(3, 32, kernel_size=(3, 3), strides=1, padding='SAME', use_bias=False, rngs=rngs)
        self.stem_bn = nnx.BatchNorm(32, rngs=rngs)
        
        # (~62x62)
        self.stage1_block1 = ResNetBlock(in_features=32, out_features=64, stride=2, rngs=rngs)
        self.stage1_block2 = ResNetBlock(in_features=64, out_features=64, stride=1, rngs=rngs)
        
        # (~31x31)
        self.stage2_block1 = ResNetBlock(in_features=64, out_features=128, stride=2, rngs=rngs)
        self.stage2_block2 = ResNetBlock(in_features=128, out_features=128, stride=1, rngs=rngs)
        
        # (15x15)
        self.stage3_block1 = ResNetBlock(in_features=128, out_features=256, stride=2, rngs=rngs)
        self.stage3_block2 = ResNetBlock(in_features=256, out_features=256, stride=1, rngs=rngs)
        
        # scoring Head
        self.head_linear1 = nnx.Linear(in_features=256, out_features=128, rngs=rngs)
        self.head_dropout = nnx.Dropout(rate=0.3, rngs=rngs)
        self.head_linear2 = nnx.Linear(in_features=128, out_features=1, rngs=rngs)

    def __call__(self, x, deterministic: bool = False, rngs: nnx.Rngs | None = None):
        
        # stem
        x = self.stem_conv(x)
        x = self.stem_bn(x, use_running_average=deterministic)
        x = nnx.relu(x)
        
        # stages
        x = self.stage1_block1(x, deterministic=deterministic)
        x = self.stage1_block2(x, deterministic=deterministic)
        
        x = self.stage2_block1(x, deterministic=deterministic)
        x = self.stage2_block2(x, deterministic=deterministic)
        
        x = self.stage3_block1(x, deterministic=deterministic)
        x = self.stage3_block2(x, deterministic=deterministic)
        
        # this collapses (batch , channel) into (bath, channel)
        x = jnp.max(x, axis=(1, 2))
        
        #  Scoring Head
        x = self.head_linear1(x)
        x = nnx.relu(x)
        x = self.head_dropout(x, deterministic=deterministic, rngs=rngs)
        x = self.head_linear2(x)
        
        return jnp.squeeze(x, axis=-1)