┐н
л¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108ўц
В
input_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameinput_98/kernel
{
#input_98/kernel/Read/ReadVariableOpReadVariableOpinput_98/kernel*&
_output_shapes
:*
dtype0
r
input_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameinput_98/bias
k
!input_98/bias/Read/ReadVariableOpReadVariableOpinput_98/bias*
_output_shapes
:*
dtype0
Д
conv_2_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_2_98/kernel
}
$conv_2_98/kernel/Read/ReadVariableOpReadVariableOpconv_2_98/kernel*&
_output_shapes
:*
dtype0
t
conv_2_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_2_98/bias
m
"conv_2_98/bias/Read/ReadVariableOpReadVariableOpconv_2_98/bias*
_output_shapes
:*
dtype0
Д
conv_3_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_3_98/kernel
}
$conv_3_98/kernel/Read/ReadVariableOpReadVariableOpconv_3_98/kernel*&
_output_shapes
:*
dtype0
t
conv_3_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3_98/bias
m
"conv_3_98/bias/Read/ReadVariableOpReadVariableOpconv_3_98/bias*
_output_shapes
:*
dtype0
}
dense1_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	М
*!
shared_namedense1_98/kernel
v
$dense1_98/kernel/Read/ReadVariableOpReadVariableOpdense1_98/kernel*
_output_shapes
:	М
*
dtype0
t
dense1_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense1_98/bias
m
"dense1_98/bias/Read/ReadVariableOpReadVariableOpdense1_98/bias*
_output_shapes
:
*
dtype0
|
output_98/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*!
shared_nameoutput_98/kernel
u
$output_98/kernel/Read/ReadVariableOpReadVariableOpoutput_98/kernel*
_output_shapes

:
*
dtype0
t
output_98/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput_98/bias
m
"output_98/bias/Read/ReadVariableOpReadVariableOpoutput_98/bias*
_output_shapes
:*
dtype0

NoOpNoOp
К
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*┼
value╗B╕ B▒
┐
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
trainable_variables
		variables

regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
 	variables
!regularization_losses
"	keras_api
h

#kernel
$bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
h

)kernel
*bias
+trainable_variables
,	variables
-regularization_losses
.	keras_api
F
0
1
2
3
4
5
#6
$7
)8
*9
F
0
1
2
3
4
5
#6
$7
)8
*9
 
Ъ
trainable_variables
		variables
/layer_regularization_losses
0non_trainable_variables

1layers
2metrics

regularization_losses
 
[Y
VARIABLE_VALUEinput_98/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEinput_98/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Ъ
trainable_variables
	variables
3layer_regularization_losses
4non_trainable_variables

5layers
6metrics
regularization_losses
\Z
VARIABLE_VALUEconv_2_98/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv_2_98/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Ъ
trainable_variables
	variables
7layer_regularization_losses
8non_trainable_variables

9layers
:metrics
regularization_losses
\Z
VARIABLE_VALUEconv_3_98/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv_3_98/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Ъ
trainable_variables
	variables
;layer_regularization_losses
<non_trainable_variables

=layers
>metrics
regularization_losses
 
 
 
Ъ
trainable_variables
 	variables
?layer_regularization_losses
@non_trainable_variables

Alayers
Bmetrics
!regularization_losses
\Z
VARIABLE_VALUEdense1_98/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense1_98/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
Ъ
%trainable_variables
&	variables
Clayer_regularization_losses
Dnon_trainable_variables

Elayers
Fmetrics
'regularization_losses
\Z
VARIABLE_VALUEoutput_98/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEoutput_98/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1

)0
*1
 
Ъ
+trainable_variables
,	variables
Glayer_regularization_losses
Hnon_trainable_variables

Ilayers
Jmetrics
-regularization_losses
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Т
serving_default_input_inputPlaceholder*1
_output_shapes
:         ╥а*
dtype0*&
shape:         ╥а
╤
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_inputinput_98/kernelinput_98/biasconv_2_98/kernelconv_2_98/biasconv_3_98/kernelconv_3_98/biasdense1_98/kerneldense1_98/biasoutput_98/kerneloutput_98/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*.
f)R'
%__inference_signature_wrapper_2830083
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
А
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#input_98/kernel/Read/ReadVariableOp!input_98/bias/Read/ReadVariableOp$conv_2_98/kernel/Read/ReadVariableOp"conv_2_98/bias/Read/ReadVariableOp$conv_3_98/kernel/Read/ReadVariableOp"conv_3_98/bias/Read/ReadVariableOp$dense1_98/kernel/Read/ReadVariableOp"dense1_98/bias/Read/ReadVariableOp$output_98/kernel/Read/ReadVariableOp"output_98/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_save_2830296
│
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_98/kernelinput_98/biasconv_2_98/kernelconv_2_98/biasconv_3_98/kernelconv_3_98/biasdense1_98/kerneldense1_98/biasoutput_98/kerneloutput_98/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference__traced_restore_2830338як
с
╓
/__inference_gen_310_ind_5_layer_call_fn_2830032
input_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinput_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_28300192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
с
╣
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2830019

inputs(
$input_statefulpartitionedcall_args_1(
$input_statefulpartitionedcall_args_2)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2)
%dense1_statefulpartitionedcall_args_1)
%dense1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identityИвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвdense1/StatefulPartitionedCallвinput/StatefulPartitionedCallвoutput/StatefulPartitionedCallж
input/StatefulPartitionedCallStatefulPartitionedCallinputs$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         ╨Ю**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_input_layer_call_and_return_conditional_losses_28298522
input/StatefulPartitionedCall╔
conv_2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         * **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_28298732 
conv_2/StatefulPartitionedCall╩
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_28298942 
conv_3/StatefulPartitionedCall╒
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         М**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_flat_layer_call_and_return_conditional_losses_28299212
flat/PartitionedCall╕
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_28299402 
dense1/StatefulPartitionedCall┬
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28299632 
output/StatefulPartitionedCallЯ
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
╫
B
&__inference_flat_layer_call_fn_2830206

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         М**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_flat_layer_call_and_return_conditional_losses_28299212
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         М2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
:& "
 
_user_specified_nameinputs
ч
█
B__inference_input_layer_call_and_return_conditional_losses_2829852

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
╛#
с
 __inference__traced_save_2830296
file_prefix.
*savev2_input_98_kernel_read_readvariableop,
(savev2_input_98_bias_read_readvariableop/
+savev2_conv_2_98_kernel_read_readvariableop-
)savev2_conv_2_98_bias_read_readvariableop/
+savev2_conv_3_98_kernel_read_readvariableop-
)savev2_conv_3_98_bias_read_readvariableop/
+savev2_dense1_98_kernel_read_readvariableop-
)savev2_dense1_98_bias_read_readvariableop/
+savev2_output_98_kernel_read_readvariableop-
)savev2_output_98_bias_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1е
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_0b64cbf99d3344bd96b2734c02f9d1ee/part2
StringJoin/inputs_1Б

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameй
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*╗
value▒Bо
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesЬ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
SaveV2/shape_and_slicesы
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_input_98_kernel_read_readvariableop(savev2_input_98_bias_read_readvariableop+savev2_conv_2_98_kernel_read_readvariableop)savev2_conv_2_98_bias_read_readvariableop+savev2_conv_3_98_kernel_read_readvariableop)savev2_conv_3_98_bias_read_readvariableop+savev2_dense1_98_kernel_read_readvariableop)savev2_dense1_98_bias_read_readvariableop+savev2_output_98_kernel_read_readvariableop)savev2_output_98_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2
2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*А
_input_shapeso
m: :::::::	М
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
╝@
╚
"__inference__wrapped_model_2829839
input_input6
2gen_310_ind_5_input_conv2d_readvariableop_resource7
3gen_310_ind_5_input_biasadd_readvariableop_resource7
3gen_310_ind_5_conv_2_conv2d_readvariableop_resource8
4gen_310_ind_5_conv_2_biasadd_readvariableop_resource7
3gen_310_ind_5_conv_3_conv2d_readvariableop_resource8
4gen_310_ind_5_conv_3_biasadd_readvariableop_resource7
3gen_310_ind_5_dense1_matmul_readvariableop_resource8
4gen_310_ind_5_dense1_biasadd_readvariableop_resource7
3gen_310_ind_5_output_matmul_readvariableop_resource8
4gen_310_ind_5_output_biasadd_readvariableop_resource
identityИв+gen_310_ind_5/conv_2/BiasAdd/ReadVariableOpв*gen_310_ind_5/conv_2/Conv2D/ReadVariableOpв+gen_310_ind_5/conv_3/BiasAdd/ReadVariableOpв*gen_310_ind_5/conv_3/Conv2D/ReadVariableOpв+gen_310_ind_5/dense1/BiasAdd/ReadVariableOpв*gen_310_ind_5/dense1/MatMul/ReadVariableOpв*gen_310_ind_5/input/BiasAdd/ReadVariableOpв)gen_310_ind_5/input/Conv2D/ReadVariableOpв+gen_310_ind_5/output/BiasAdd/ReadVariableOpв*gen_310_ind_5/output/MatMul/ReadVariableOp╤
)gen_310_ind_5/input/Conv2D/ReadVariableOpReadVariableOp2gen_310_ind_5_input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)gen_310_ind_5/input/Conv2D/ReadVariableOpч
gen_310_ind_5/input/Conv2DConv2Dinput_input1gen_310_ind_5/input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╨Ю*
paddingVALID*
strides
2
gen_310_ind_5/input/Conv2D╚
*gen_310_ind_5/input/BiasAdd/ReadVariableOpReadVariableOp3gen_310_ind_5_input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*gen_310_ind_5/input/BiasAdd/ReadVariableOp┌
gen_310_ind_5/input/BiasAddBiasAdd#gen_310_ind_5/input/Conv2D:output:02gen_310_ind_5/input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╨Ю2
gen_310_ind_5/input/BiasAddЮ
gen_310_ind_5/input/ReluRelu$gen_310_ind_5/input/BiasAdd:output:0*
T0*1
_output_shapes
:         ╨Ю2
gen_310_ind_5/input/Relu╘
*gen_310_ind_5/conv_2/Conv2D/ReadVariableOpReadVariableOp3gen_310_ind_5_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*gen_310_ind_5/conv_2/Conv2D/ReadVariableOpГ
gen_310_ind_5/conv_2/Conv2DConv2D&gen_310_ind_5/input/Relu:activations:02gen_310_ind_5/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         * *
paddingVALID*
strides
2
gen_310_ind_5/conv_2/Conv2D╦
+gen_310_ind_5/conv_2/BiasAdd/ReadVariableOpReadVariableOp4gen_310_ind_5_conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+gen_310_ind_5/conv_2/BiasAdd/ReadVariableOp▄
gen_310_ind_5/conv_2/BiasAddBiasAdd$gen_310_ind_5/conv_2/Conv2D:output:03gen_310_ind_5/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         * 2
gen_310_ind_5/conv_2/BiasAddЯ
gen_310_ind_5/conv_2/ReluRelu%gen_310_ind_5/conv_2/BiasAdd:output:0*
T0*/
_output_shapes
:         * 2
gen_310_ind_5/conv_2/Relu╘
*gen_310_ind_5/conv_3/Conv2D/ReadVariableOpReadVariableOp3gen_310_ind_5_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*gen_310_ind_5/conv_3/Conv2D/ReadVariableOpД
gen_310_ind_5/conv_3/Conv2DConv2D'gen_310_ind_5/conv_2/Relu:activations:02gen_310_ind_5/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
gen_310_ind_5/conv_3/Conv2D╦
+gen_310_ind_5/conv_3/BiasAdd/ReadVariableOpReadVariableOp4gen_310_ind_5_conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+gen_310_ind_5/conv_3/BiasAdd/ReadVariableOp▄
gen_310_ind_5/conv_3/BiasAddBiasAdd$gen_310_ind_5/conv_3/Conv2D:output:03gen_310_ind_5/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
gen_310_ind_5/conv_3/BiasAddЯ
gen_310_ind_5/conv_3/ReluRelu%gen_310_ind_5/conv_3/BiasAdd:output:0*
T0*/
_output_shapes
:         
2
gen_310_ind_5/conv_3/ReluЕ
gen_310_ind_5/flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"    М   2
gen_310_ind_5/flat/Const┬
gen_310_ind_5/flat/ReshapeReshape'gen_310_ind_5/conv_3/Relu:activations:0!gen_310_ind_5/flat/Const:output:0*
T0*(
_output_shapes
:         М2
gen_310_ind_5/flat/Reshape═
*gen_310_ind_5/dense1/MatMul/ReadVariableOpReadVariableOp3gen_310_ind_5_dense1_matmul_readvariableop_resource*
_output_shapes
:	М
*
dtype02,
*gen_310_ind_5/dense1/MatMul/ReadVariableOp╧
gen_310_ind_5/dense1/MatMulMatMul#gen_310_ind_5/flat/Reshape:output:02gen_310_ind_5/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
gen_310_ind_5/dense1/MatMul╦
+gen_310_ind_5/dense1/BiasAdd/ReadVariableOpReadVariableOp4gen_310_ind_5_dense1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+gen_310_ind_5/dense1/BiasAdd/ReadVariableOp╒
gen_310_ind_5/dense1/BiasAddBiasAdd%gen_310_ind_5/dense1/MatMul:product:03gen_310_ind_5/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
gen_310_ind_5/dense1/BiasAddа
gen_310_ind_5/dense1/SigmoidSigmoid%gen_310_ind_5/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
gen_310_ind_5/dense1/Sigmoid╠
*gen_310_ind_5/output/MatMul/ReadVariableOpReadVariableOp3gen_310_ind_5_output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*gen_310_ind_5/output/MatMul/ReadVariableOp╠
gen_310_ind_5/output/MatMulMatMul gen_310_ind_5/dense1/Sigmoid:y:02gen_310_ind_5/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
gen_310_ind_5/output/MatMul╦
+gen_310_ind_5/output/BiasAdd/ReadVariableOpReadVariableOp4gen_310_ind_5_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+gen_310_ind_5/output/BiasAdd/ReadVariableOp╒
gen_310_ind_5/output/BiasAddBiasAdd%gen_310_ind_5/output/MatMul:product:03gen_310_ind_5/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
gen_310_ind_5/output/BiasAddа
gen_310_ind_5/output/SoftmaxSoftmax%gen_310_ind_5/output/BiasAdd:output:0*
T0*'
_output_shapes
:         2
gen_310_ind_5/output/Softmax┐
IdentityIdentity&gen_310_ind_5/output/Softmax:softmax:0,^gen_310_ind_5/conv_2/BiasAdd/ReadVariableOp+^gen_310_ind_5/conv_2/Conv2D/ReadVariableOp,^gen_310_ind_5/conv_3/BiasAdd/ReadVariableOp+^gen_310_ind_5/conv_3/Conv2D/ReadVariableOp,^gen_310_ind_5/dense1/BiasAdd/ReadVariableOp+^gen_310_ind_5/dense1/MatMul/ReadVariableOp+^gen_310_ind_5/input/BiasAdd/ReadVariableOp*^gen_310_ind_5/input/Conv2D/ReadVariableOp,^gen_310_ind_5/output/BiasAdd/ReadVariableOp+^gen_310_ind_5/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::2Z
+gen_310_ind_5/conv_2/BiasAdd/ReadVariableOp+gen_310_ind_5/conv_2/BiasAdd/ReadVariableOp2X
*gen_310_ind_5/conv_2/Conv2D/ReadVariableOp*gen_310_ind_5/conv_2/Conv2D/ReadVariableOp2Z
+gen_310_ind_5/conv_3/BiasAdd/ReadVariableOp+gen_310_ind_5/conv_3/BiasAdd/ReadVariableOp2X
*gen_310_ind_5/conv_3/Conv2D/ReadVariableOp*gen_310_ind_5/conv_3/Conv2D/ReadVariableOp2Z
+gen_310_ind_5/dense1/BiasAdd/ReadVariableOp+gen_310_ind_5/dense1/BiasAdd/ReadVariableOp2X
*gen_310_ind_5/dense1/MatMul/ReadVariableOp*gen_310_ind_5/dense1/MatMul/ReadVariableOp2X
*gen_310_ind_5/input/BiasAdd/ReadVariableOp*gen_310_ind_5/input/BiasAdd/ReadVariableOp2V
)gen_310_ind_5/input/Conv2D/ReadVariableOp)gen_310_ind_5/input/Conv2D/ReadVariableOp2Z
+gen_310_ind_5/output/BiasAdd/ReadVariableOp+gen_310_ind_5/output/BiasAdd/ReadVariableOp2X
*gen_310_ind_5/output/MatMul/ReadVariableOp*gen_310_ind_5/output/MatMul/ReadVariableOp:+ '
%
_user_specified_nameinput_input
╝
и
'__inference_input_layer_call_fn_2829860

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           **
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_input_layer_call_and_return_conditional_losses_28298522
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╚	
▄
C__inference_dense1_layer_call_and_return_conditional_losses_2830217

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	М
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         М::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ё
╛
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2829976
input_input(
$input_statefulpartitionedcall_args_1(
$input_statefulpartitionedcall_args_2)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2)
%dense1_statefulpartitionedcall_args_1)
%dense1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identityИвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвdense1/StatefulPartitionedCallвinput/StatefulPartitionedCallвoutput/StatefulPartitionedCallл
input/StatefulPartitionedCallStatefulPartitionedCallinput_input$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         ╨Ю**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_input_layer_call_and_return_conditional_losses_28298522
input/StatefulPartitionedCall╔
conv_2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         * **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_28298732 
conv_2/StatefulPartitionedCall╩
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_28298942 
conv_3/StatefulPartitionedCall╒
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         М**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_flat_layer_call_and_return_conditional_losses_28299212
flat/PartitionedCall╕
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_28299402 
dense1/StatefulPartitionedCall┬
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28299632 
output/StatefulPartitionedCallЯ
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
╠	
▄
C__inference_output_layer_call_and_return_conditional_losses_2830235

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Щ1
╙
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2830124

inputs(
$input_conv2d_readvariableop_resource)
%input_biasadd_readvariableop_resource)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityИвconv_2/BiasAdd/ReadVariableOpвconv_2/Conv2D/ReadVariableOpвconv_3/BiasAdd/ReadVariableOpвconv_3/Conv2D/ReadVariableOpвdense1/BiasAdd/ReadVariableOpвdense1/MatMul/ReadVariableOpвinput/BiasAdd/ReadVariableOpвinput/Conv2D/ReadVariableOpвoutput/BiasAdd/ReadVariableOpвoutput/MatMul/ReadVariableOpз
input/Conv2D/ReadVariableOpReadVariableOp$input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
input/Conv2D/ReadVariableOp╕
input/Conv2DConv2Dinputs#input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╨Ю*
paddingVALID*
strides
2
input/Conv2DЮ
input/BiasAdd/ReadVariableOpReadVariableOp%input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
input/BiasAdd/ReadVariableOpв
input/BiasAddBiasAddinput/Conv2D:output:0$input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╨Ю2
input/BiasAddt

input/ReluReluinput/BiasAdd:output:0*
T0*1
_output_shapes
:         ╨Ю2

input/Reluк
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_2/Conv2D/ReadVariableOp╦
conv_2/Conv2DConv2Dinput/Relu:activations:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         * *
paddingVALID*
strides
2
conv_2/Conv2Dб
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_2/BiasAdd/ReadVariableOpд
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         * 2
conv_2/BiasAddu
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*/
_output_shapes
:         * 2
conv_2/Reluк
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_3/Conv2D/ReadVariableOp╠
conv_3/Conv2DConv2Dconv_2/Relu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
conv_3/Conv2Dб
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOpд
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
conv_3/BiasAddu
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*/
_output_shapes
:         
2
conv_3/Relui

flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"    М   2

flat/ConstК
flat/ReshapeReshapeconv_3/Relu:activations:0flat/Const:output:0*
T0*(
_output_shapes
:         М2
flat/Reshapeг
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	М
*
dtype02
dense1/MatMul/ReadVariableOpЧ
dense1/MatMulMatMulflat/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense1/MatMulб
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense1/BiasAdd/ReadVariableOpЭ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense1/BiasAddv
dense1/SigmoidSigmoiddense1/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense1/Sigmoidв
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
output/MatMul/ReadVariableOpФ
output/MatMulMatMuldense1/Sigmoid:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulб
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:         2
output/Softmaxе
IdentityIdentityoutput/Softmax:softmax:0^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^input/BiasAdd/ReadVariableOp^input/Conv2D/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2<
input/BiasAdd/ReadVariableOpinput/BiasAdd/ReadVariableOp2:
input/Conv2D/ReadVariableOpinput/Conv2D/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
с
╓
/__inference_gen_310_ind_5_layer_call_fn_2830067
input_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinput_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_28300542
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
с
╣
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2830054

inputs(
$input_statefulpartitionedcall_args_1(
$input_statefulpartitionedcall_args_2)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2)
%dense1_statefulpartitionedcall_args_1)
%dense1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identityИвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвdense1/StatefulPartitionedCallвinput/StatefulPartitionedCallвoutput/StatefulPartitionedCallж
input/StatefulPartitionedCallStatefulPartitionedCallinputs$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         ╨Ю**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_input_layer_call_and_return_conditional_losses_28298522
input/StatefulPartitionedCall╔
conv_2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         * **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_28298732 
conv_2/StatefulPartitionedCall╩
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_28298942 
conv_3/StatefulPartitionedCall╒
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         М**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_flat_layer_call_and_return_conditional_losses_28299212
flat/PartitionedCall╕
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_28299402 
dense1/StatefulPartitionedCall┬
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28299632 
output/StatefulPartitionedCallЯ
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ш
▄
C__inference_conv_2_layer_call_and_return_conditional_losses_2829873

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
И
]
A__inference_flat_layer_call_and_return_conditional_losses_2830201

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    М   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         М2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         М2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
:& "
 
_user_specified_nameinputs
п
╠
%__inference_signature_wrapper_2830083
input_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinput_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*+
f&R$
"__inference__wrapped_model_28298392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
╛
й
(__inference_conv_2_layer_call_fn_2829881

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_28298732
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╥
╤
/__inference_gen_310_ind_5_layer_call_fn_2830195

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИвStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_28300542
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Щ1
╙
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2830165

inputs(
$input_conv2d_readvariableop_resource)
%input_biasadd_readvariableop_resource)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityИвconv_2/BiasAdd/ReadVariableOpвconv_2/Conv2D/ReadVariableOpвconv_3/BiasAdd/ReadVariableOpвconv_3/Conv2D/ReadVariableOpвdense1/BiasAdd/ReadVariableOpвdense1/MatMul/ReadVariableOpвinput/BiasAdd/ReadVariableOpвinput/Conv2D/ReadVariableOpвoutput/BiasAdd/ReadVariableOpвoutput/MatMul/ReadVariableOpз
input/Conv2D/ReadVariableOpReadVariableOp$input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
input/Conv2D/ReadVariableOp╕
input/Conv2DConv2Dinputs#input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╨Ю*
paddingVALID*
strides
2
input/Conv2DЮ
input/BiasAdd/ReadVariableOpReadVariableOp%input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
input/BiasAdd/ReadVariableOpв
input/BiasAddBiasAddinput/Conv2D:output:0$input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╨Ю2
input/BiasAddt

input/ReluReluinput/BiasAdd:output:0*
T0*1
_output_shapes
:         ╨Ю2

input/Reluк
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_2/Conv2D/ReadVariableOp╦
conv_2/Conv2DConv2Dinput/Relu:activations:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         * *
paddingVALID*
strides
2
conv_2/Conv2Dб
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_2/BiasAdd/ReadVariableOpд
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         * 2
conv_2/BiasAddu
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*/
_output_shapes
:         * 2
conv_2/Reluк
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_3/Conv2D/ReadVariableOp╠
conv_3/Conv2DConv2Dconv_2/Relu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
conv_3/Conv2Dб
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOpд
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
conv_3/BiasAddu
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*/
_output_shapes
:         
2
conv_3/Relui

flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"    М   2

flat/ConstК
flat/ReshapeReshapeconv_3/Relu:activations:0flat/Const:output:0*
T0*(
_output_shapes
:         М2
flat/Reshapeг
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	М
*
dtype02
dense1/MatMul/ReadVariableOpЧ
dense1/MatMulMatMulflat/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense1/MatMulб
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense1/BiasAdd/ReadVariableOpЭ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense1/BiasAddv
dense1/SigmoidSigmoiddense1/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense1/Sigmoidв
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
output/MatMul/ReadVariableOpФ
output/MatMulMatMuldense1/Sigmoid:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/MatMulб
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЭ
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:         2
output/Softmaxе
IdentityIdentityoutput/Softmax:softmax:0^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^input/BiasAdd/ReadVariableOp^input/Conv2D/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2<
input/BiasAdd/ReadVariableOpinput/BiasAdd/ReadVariableOp2:
input/Conv2D/ReadVariableOpinput/Conv2D/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ш
▄
C__inference_conv_3_layer_call_and_return_conditional_losses_2829894

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp╢
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           2
Relu▒
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
И
]
A__inference_flat_layer_call_and_return_conditional_losses_2829921

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    М   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         М2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         М2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
:& "
 
_user_specified_nameinputs
╥
╤
/__inference_gen_310_ind_5_layer_call_fn_2830180

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИвStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_28300192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╚	
▄
C__inference_dense1_layer_call_and_return_conditional_losses_2829940

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	М
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         М::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
╓/
└
#__inference__traced_restore_2830338
file_prefix$
 assignvariableop_input_98_kernel$
 assignvariableop_1_input_98_bias'
#assignvariableop_2_conv_2_98_kernel%
!assignvariableop_3_conv_2_98_bias'
#assignvariableop_4_conv_3_98_kernel%
!assignvariableop_5_conv_3_98_bias'
#assignvariableop_6_dense1_98_kernel%
!assignvariableop_7_dense1_98_bias'
#assignvariableop_8_output_98_kernel%
!assignvariableop_9_output_98_bias
identity_11ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1п
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*╗
value▒Bо
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesв
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
RestoreV2/shape_and_slices▌
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityР
AssignVariableOpAssignVariableOp assignvariableop_input_98_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ц
AssignVariableOp_1AssignVariableOp assignvariableop_1_input_98_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Щ
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv_2_98_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ч
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv_2_98_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Щ
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv_3_98_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Ч
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv_3_98_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Щ
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense1_98_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ч
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense1_98_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Щ
AssignVariableOp_8AssignVariableOp#assignvariableop_8_output_98_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ч
AssignVariableOp_9AssignVariableOp!assignvariableop_9_output_98_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp║
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10╟
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
╠	
▄
C__inference_output_layer_call_and_return_conditional_losses_2829963

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ё
й
(__inference_dense1_layer_call_fn_2830224

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_28299402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         М::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╛
й
(__inference_conv_3_layer_call_fn_2829902

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+                           **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_28298942
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ё
╛
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2829996
input_input(
$input_statefulpartitionedcall_args_1(
$input_statefulpartitionedcall_args_2)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2)
%dense1_statefulpartitionedcall_args_1)
%dense1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identityИвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвdense1/StatefulPartitionedCallвinput/StatefulPartitionedCallвoutput/StatefulPartitionedCallл
input/StatefulPartitionedCallStatefulPartitionedCallinput_input$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:         ╨Ю**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_input_layer_call_and_return_conditional_losses_28298522
input/StatefulPartitionedCall╔
conv_2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         * **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv_2_layer_call_and_return_conditional_losses_28298732 
conv_2/StatefulPartitionedCall╩
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:         
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv_3_layer_call_and_return_conditional_losses_28298942 
conv_3/StatefulPartitionedCall╒
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:         М**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_flat_layer_call_and_return_conditional_losses_28299212
flat/PartitionedCall╕
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         
**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_28299402 
dense1/StatefulPartitionedCall┬
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28299632 
output/StatefulPartitionedCallЯ
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
Ё
й
(__inference_output_layer_call_fn_2830242

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_28299632
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         
::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╗
serving_defaultз
M
input_input>
serving_default_input_input:0         ╥а:
output0
StatefulPartitionedCall:0         tensorflow/serving/predict:е┐
°0
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
trainable_variables
		variables

regularization_losses
	keras_api

signatures
K__call__
*L&call_and_return_all_conditional_losses
M_default_save_signature"▀-
_tf_keras_sequential└-{"class_name": "Sequential", "name": "gen_310_ind_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gen_310_ind_5", "layers": [{"class_name": "Conv2D", "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "filters": 6, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [5, 5], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "gen_310_ind_5", "layers": [{"class_name": "Conv2D", "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "filters": 6, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [5, 5], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
╣"╢
_tf_keras_input_layerЦ{"class_name": "InputLayer", "name": "input_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 210, 160, 4], "config": {"batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_input"}}
Ю

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
N__call__
*O&call_and_return_all_conditional_losses"∙
_tf_keras_layer▀{"class_name": "Conv2D", "name": "input", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 210, 160, 4], "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "filters": 6, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}}
ч

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"┬
_tf_keras_layerи{"class_name": "Conv2D", "name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [5, 5], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 6}}}}
ч

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"┬
_tf_keras_layerи{"class_name": "Conv2D", "name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
ж
trainable_variables
 	variables
!regularization_losses
"	keras_api
T__call__
*U&call_and_return_all_conditional_losses"Ч
_tf_keras_layer¤{"class_name": "Flatten", "name": "flat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
є

#kernel
$bias
%trainable_variables
&	variables
'regularization_losses
(	keras_api
V__call__
*W&call_and_return_all_conditional_losses"╬
_tf_keras_layer┤{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 140}}}}
ё

)kernel
*bias
+trainable_variables
,	variables
-regularization_losses
.	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"╠
_tf_keras_layer▓{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}}
f
0
1
2
3
4
5
#6
$7
)8
*9"
trackable_list_wrapper
f
0
1
2
3
4
5
#6
$7
)8
*9"
trackable_list_wrapper
 "
trackable_list_wrapper
╖
trainable_variables
		variables
/layer_regularization_losses
0non_trainable_variables

1layers
2metrics

regularization_losses
K__call__
M_default_save_signature
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
,
Zserving_default"
signature_map
):'2input_98/kernel
:2input_98/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
trainable_variables
	variables
3layer_regularization_losses
4non_trainable_variables

5layers
6metrics
regularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
*:(2conv_2_98/kernel
:2conv_2_98/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
trainable_variables
	variables
7layer_regularization_losses
8non_trainable_variables

9layers
:metrics
regularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
*:(2conv_3_98/kernel
:2conv_3_98/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
trainable_variables
	variables
;layer_regularization_losses
<non_trainable_variables

=layers
>metrics
regularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
trainable_variables
 	variables
?layer_regularization_losses
@non_trainable_variables

Alayers
Bmetrics
!regularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
#:!	М
2dense1_98/kernel
:
2dense1_98/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
%trainable_variables
&	variables
Clayer_regularization_losses
Dnon_trainable_variables

Elayers
Fmetrics
'regularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
": 
2output_98/kernel
:2output_98/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
+trainable_variables
,	variables
Glayer_regularization_losses
Hnon_trainable_variables

Ilayers
Jmetrics
-regularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
К2З
/__inference_gen_310_ind_5_layer_call_fn_2830180
/__inference_gen_310_ind_5_layer_call_fn_2830032
/__inference_gen_310_ind_5_layer_call_fn_2830195
/__inference_gen_310_ind_5_layer_call_fn_2830067└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ў2є
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2830124
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2829976
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2830165
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2829996└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ю2ы
"__inference__wrapped_model_2829839─
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *4в1
/К,
input_input         ╥а
Ж2Г
'__inference_input_layer_call_fn_2829860╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
б2Ю
B__inference_input_layer_call_and_return_conditional_losses_2829852╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
З2Д
(__inference_conv_2_layer_call_fn_2829881╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
в2Я
C__inference_conv_2_layer_call_and_return_conditional_losses_2829873╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
З2Д
(__inference_conv_3_layer_call_fn_2829902╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
в2Я
C__inference_conv_3_layer_call_and_return_conditional_losses_2829894╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
╨2═
&__inference_flat_layer_call_fn_2830206в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_flat_layer_call_and_return_conditional_losses_2830201в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense1_layer_call_fn_2830224в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense1_layer_call_and_return_conditional_losses_2830217в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_output_layer_call_fn_2830242в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_output_layer_call_and_return_conditional_losses_2830235в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
8B6
%__inference_signature_wrapper_2830083input_inputг
"__inference__wrapped_model_2829839}
#$)*>в;
4в1
/К,
input_input         ╥а
к "/к,
*
output К
output         ╪
C__inference_conv_2_layer_call_and_return_conditional_losses_2829873РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ░
(__inference_conv_2_layer_call_fn_2829881ГIвF
?в<
:К7
inputs+                           
к "2К/+                           ╪
C__inference_conv_3_layer_call_and_return_conditional_losses_2829894РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ░
(__inference_conv_3_layer_call_fn_2829902ГIвF
?в<
:К7
inputs+                           
к "2К/+                           д
C__inference_dense1_layer_call_and_return_conditional_losses_2830217]#$0в-
&в#
!К
inputs         М
к "%в"
К
0         

Ъ |
(__inference_dense1_layer_call_fn_2830224P#$0в-
&в#
!К
inputs         М
к "К         
ж
A__inference_flat_layer_call_and_return_conditional_losses_2830201a7в4
-в*
(К%
inputs         

к "&в#
К
0         М
Ъ ~
&__inference_flat_layer_call_fn_2830206T7в4
-в*
(К%
inputs         

к "К         М╔
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2829976{
#$)*FвC
<в9
/К,
input_input         ╥а
p

 
к "%в"
К
0         
Ъ ╔
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2829996{
#$)*FвC
<в9
/К,
input_input         ╥а
p 

 
к "%в"
К
0         
Ъ ─
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2830124v
#$)*Aв>
7в4
*К'
inputs         ╥а
p

 
к "%в"
К
0         
Ъ ─
J__inference_gen_310_ind_5_layer_call_and_return_conditional_losses_2830165v
#$)*Aв>
7в4
*К'
inputs         ╥а
p 

 
к "%в"
К
0         
Ъ б
/__inference_gen_310_ind_5_layer_call_fn_2830032n
#$)*FвC
<в9
/К,
input_input         ╥а
p

 
к "К         б
/__inference_gen_310_ind_5_layer_call_fn_2830067n
#$)*FвC
<в9
/К,
input_input         ╥а
p 

 
к "К         Ь
/__inference_gen_310_ind_5_layer_call_fn_2830180i
#$)*Aв>
7в4
*К'
inputs         ╥а
p

 
к "К         Ь
/__inference_gen_310_ind_5_layer_call_fn_2830195i
#$)*Aв>
7в4
*К'
inputs         ╥а
p 

 
к "К         ╫
B__inference_input_layer_call_and_return_conditional_losses_2829852РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ п
'__inference_input_layer_call_fn_2829860ГIвF
?в<
:К7
inputs+                           
к "2К/+                           г
C__inference_output_layer_call_and_return_conditional_losses_2830235\)*/в,
%в"
 К
inputs         

к "%в"
К
0         
Ъ {
(__inference_output_layer_call_fn_2830242O)*/в,
%в"
 К
inputs         

к "К         ╢
%__inference_signature_wrapper_2830083М
#$)*MвJ
в 
Cк@
>
input_input/К,
input_input         ╥а"/к,
*
output К
output         