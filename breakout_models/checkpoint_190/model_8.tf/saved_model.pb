��
��
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
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108��
�
input_602/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameinput_602/kernel
}
$input_602/kernel/Read/ReadVariableOpReadVariableOpinput_602/kernel*&
_output_shapes
:*
dtype0
t
input_602/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameinput_602/bias
m
"input_602/bias/Read/ReadVariableOpReadVariableOpinput_602/bias*
_output_shapes
:*
dtype0
�
conv_2_602/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv_2_602/kernel

%conv_2_602/kernel/Read/ReadVariableOpReadVariableOpconv_2_602/kernel*&
_output_shapes
:*
dtype0
v
conv_2_602/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv_2_602/bias
o
#conv_2_602/bias/Read/ReadVariableOpReadVariableOpconv_2_602/bias*
_output_shapes
:*
dtype0
�
conv_3_602/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv_3_602/kernel

%conv_3_602/kernel/Read/ReadVariableOpReadVariableOpconv_3_602/kernel*&
_output_shapes
:*
dtype0
v
conv_3_602/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv_3_602/bias
o
#conv_3_602/bias/Read/ReadVariableOpReadVariableOpconv_3_602/bias*
_output_shapes
:*
dtype0

dense1_602/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*"
shared_namedense1_602/kernel
x
%dense1_602/kernel/Read/ReadVariableOpReadVariableOpdense1_602/kernel*
_output_shapes
:	�
*
dtype0
v
dense1_602/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense1_602/bias
o
#dense1_602/bias/Read/ReadVariableOpReadVariableOpdense1_602/bias*
_output_shapes
:
*
dtype0
~
output_602/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*"
shared_nameoutput_602/kernel
w
%output_602/kernel/Read/ReadVariableOpReadVariableOpoutput_602/kernel*
_output_shapes

:
*
dtype0
v
output_602/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameoutput_602/bias
o
#output_602/bias/Read/ReadVariableOpReadVariableOpoutput_602/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
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
regularization_losses
		variables

trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
 	variables
!trainable_variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
 
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
�
/non_trainable_variables

0layers
1layer_regularization_losses
regularization_losses
2metrics
		variables

trainable_variables
 
\Z
VARIABLE_VALUEinput_602/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEinput_602/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
3non_trainable_variables

4layers
5layer_regularization_losses
regularization_losses
6metrics
	variables
trainable_variables
][
VARIABLE_VALUEconv_2_602/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv_2_602/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
7non_trainable_variables

8layers
9layer_regularization_losses
regularization_losses
:metrics
	variables
trainable_variables
][
VARIABLE_VALUEconv_3_602/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv_3_602/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
;non_trainable_variables

<layers
=layer_regularization_losses
regularization_losses
>metrics
	variables
trainable_variables
 
 
 
�
?non_trainable_variables

@layers
Alayer_regularization_losses
regularization_losses
Bmetrics
 	variables
!trainable_variables
][
VARIABLE_VALUEdense1_602/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense1_602/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
�
Cnon_trainable_variables

Dlayers
Elayer_regularization_losses
%regularization_losses
Fmetrics
&	variables
'trainable_variables
][
VARIABLE_VALUEoutput_602/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEoutput_602/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
�
Gnon_trainable_variables

Hlayers
Ilayer_regularization_losses
+regularization_losses
Jmetrics
,	variables
-trainable_variables
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
 
�
serving_default_input_inputPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_inputinput_602/kernelinput_602/biasconv_2_602/kernelconv_2_602/biasconv_3_602/kernelconv_3_602/biasdense1_602/kerneldense1_602/biasoutput_602/kerneloutput_602/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

GPU 

CPU2J 8*/
f*R(
&__inference_signature_wrapper_18289278
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$input_602/kernel/Read/ReadVariableOp"input_602/bias/Read/ReadVariableOp%conv_2_602/kernel/Read/ReadVariableOp#conv_2_602/bias/Read/ReadVariableOp%conv_3_602/kernel/Read/ReadVariableOp#conv_3_602/bias/Read/ReadVariableOp%dense1_602/kernel/Read/ReadVariableOp#dense1_602/bias/Read/ReadVariableOp%output_602/kernel/Read/ReadVariableOp#output_602/bias/Read/ReadVariableOpConst*
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
CPU2J 8**
f%R#
!__inference__traced_save_18289491
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_602/kernelinput_602/biasconv_2_602/kernelconv_2_602/biasconv_3_602/kernelconv_3_602/biasdense1_602/kerneldense1_602/biasoutput_602/kerneloutput_602/bias*
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
CPU2J 8*-
f(R&
$__inference__traced_restore_18289533׫
�
�
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289214

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
identity��conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�dense1/StatefulPartitionedCall�input/StatefulPartitionedCall�output/StatefulPartitionedCall�
input/StatefulPartitionedCallStatefulPartitionedCallinputs$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_182890472
input/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������* **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_182890682 
conv_2/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_182890892 
conv_3/StatefulPartitionedCall�
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_182891162
flat/PartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_182891352 
dense1/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_182891582 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�	
�
D__inference_dense1_layer_call_and_return_conditional_losses_18289135

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
0__inference_gen_190_ind_5_layer_call_fn_18289262
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_182892492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
�	
�
D__inference_output_layer_call_and_return_conditional_losses_18289430

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289171
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
identity��conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�dense1/StatefulPartitionedCall�input/StatefulPartitionedCall�output/StatefulPartitionedCall�
input/StatefulPartitionedCallStatefulPartitionedCallinput_input$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_182890472
input/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������* **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_182890682 
conv_2/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_182890892 
conv_3/StatefulPartitionedCall�
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_182891162
flat/PartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_182891352 
dense1/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_182891582 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
�
�
)__inference_conv_3_layer_call_fn_18289097

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_182890892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
0__inference_gen_190_ind_5_layer_call_fn_18289227
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_182892142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
�@
�
#__inference__wrapped_model_18289034
input_input6
2gen_190_ind_5_input_conv2d_readvariableop_resource7
3gen_190_ind_5_input_biasadd_readvariableop_resource7
3gen_190_ind_5_conv_2_conv2d_readvariableop_resource8
4gen_190_ind_5_conv_2_biasadd_readvariableop_resource7
3gen_190_ind_5_conv_3_conv2d_readvariableop_resource8
4gen_190_ind_5_conv_3_biasadd_readvariableop_resource7
3gen_190_ind_5_dense1_matmul_readvariableop_resource8
4gen_190_ind_5_dense1_biasadd_readvariableop_resource7
3gen_190_ind_5_output_matmul_readvariableop_resource8
4gen_190_ind_5_output_biasadd_readvariableop_resource
identity��+gen_190_ind_5/conv_2/BiasAdd/ReadVariableOp�*gen_190_ind_5/conv_2/Conv2D/ReadVariableOp�+gen_190_ind_5/conv_3/BiasAdd/ReadVariableOp�*gen_190_ind_5/conv_3/Conv2D/ReadVariableOp�+gen_190_ind_5/dense1/BiasAdd/ReadVariableOp�*gen_190_ind_5/dense1/MatMul/ReadVariableOp�*gen_190_ind_5/input/BiasAdd/ReadVariableOp�)gen_190_ind_5/input/Conv2D/ReadVariableOp�+gen_190_ind_5/output/BiasAdd/ReadVariableOp�*gen_190_ind_5/output/MatMul/ReadVariableOp�
)gen_190_ind_5/input/Conv2D/ReadVariableOpReadVariableOp2gen_190_ind_5_input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)gen_190_ind_5/input/Conv2D/ReadVariableOp�
gen_190_ind_5/input/Conv2DConv2Dinput_input1gen_190_ind_5/input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
2
gen_190_ind_5/input/Conv2D�
*gen_190_ind_5/input/BiasAdd/ReadVariableOpReadVariableOp3gen_190_ind_5_input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*gen_190_ind_5/input/BiasAdd/ReadVariableOp�
gen_190_ind_5/input/BiasAddBiasAdd#gen_190_ind_5/input/Conv2D:output:02gen_190_ind_5/input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
gen_190_ind_5/input/BiasAdd�
gen_190_ind_5/input/ReluRelu$gen_190_ind_5/input/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2
gen_190_ind_5/input/Relu�
*gen_190_ind_5/conv_2/Conv2D/ReadVariableOpReadVariableOp3gen_190_ind_5_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*gen_190_ind_5/conv_2/Conv2D/ReadVariableOp�
gen_190_ind_5/conv_2/Conv2DConv2D&gen_190_ind_5/input/Relu:activations:02gen_190_ind_5/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������* *
paddingVALID*
strides
2
gen_190_ind_5/conv_2/Conv2D�
+gen_190_ind_5/conv_2/BiasAdd/ReadVariableOpReadVariableOp4gen_190_ind_5_conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+gen_190_ind_5/conv_2/BiasAdd/ReadVariableOp�
gen_190_ind_5/conv_2/BiasAddBiasAdd$gen_190_ind_5/conv_2/Conv2D:output:03gen_190_ind_5/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������* 2
gen_190_ind_5/conv_2/BiasAdd�
gen_190_ind_5/conv_2/ReluRelu%gen_190_ind_5/conv_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������* 2
gen_190_ind_5/conv_2/Relu�
*gen_190_ind_5/conv_3/Conv2D/ReadVariableOpReadVariableOp3gen_190_ind_5_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*gen_190_ind_5/conv_3/Conv2D/ReadVariableOp�
gen_190_ind_5/conv_3/Conv2DConv2D'gen_190_ind_5/conv_2/Relu:activations:02gen_190_ind_5/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
gen_190_ind_5/conv_3/Conv2D�
+gen_190_ind_5/conv_3/BiasAdd/ReadVariableOpReadVariableOp4gen_190_ind_5_conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+gen_190_ind_5/conv_3/BiasAdd/ReadVariableOp�
gen_190_ind_5/conv_3/BiasAddBiasAdd$gen_190_ind_5/conv_3/Conv2D:output:03gen_190_ind_5/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
2
gen_190_ind_5/conv_3/BiasAdd�
gen_190_ind_5/conv_3/ReluRelu%gen_190_ind_5/conv_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������
2
gen_190_ind_5/conv_3/Relu�
gen_190_ind_5/flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
gen_190_ind_5/flat/Const�
gen_190_ind_5/flat/ReshapeReshape'gen_190_ind_5/conv_3/Relu:activations:0!gen_190_ind_5/flat/Const:output:0*
T0*(
_output_shapes
:����������2
gen_190_ind_5/flat/Reshape�
*gen_190_ind_5/dense1/MatMul/ReadVariableOpReadVariableOp3gen_190_ind_5_dense1_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02,
*gen_190_ind_5/dense1/MatMul/ReadVariableOp�
gen_190_ind_5/dense1/MatMulMatMul#gen_190_ind_5/flat/Reshape:output:02gen_190_ind_5/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
gen_190_ind_5/dense1/MatMul�
+gen_190_ind_5/dense1/BiasAdd/ReadVariableOpReadVariableOp4gen_190_ind_5_dense1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+gen_190_ind_5/dense1/BiasAdd/ReadVariableOp�
gen_190_ind_5/dense1/BiasAddBiasAdd%gen_190_ind_5/dense1/MatMul:product:03gen_190_ind_5/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
gen_190_ind_5/dense1/BiasAdd�
gen_190_ind_5/dense1/SigmoidSigmoid%gen_190_ind_5/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
gen_190_ind_5/dense1/Sigmoid�
*gen_190_ind_5/output/MatMul/ReadVariableOpReadVariableOp3gen_190_ind_5_output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*gen_190_ind_5/output/MatMul/ReadVariableOp�
gen_190_ind_5/output/MatMulMatMul gen_190_ind_5/dense1/Sigmoid:y:02gen_190_ind_5/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
gen_190_ind_5/output/MatMul�
+gen_190_ind_5/output/BiasAdd/ReadVariableOpReadVariableOp4gen_190_ind_5_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+gen_190_ind_5/output/BiasAdd/ReadVariableOp�
gen_190_ind_5/output/BiasAddBiasAdd%gen_190_ind_5/output/MatMul:product:03gen_190_ind_5/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
gen_190_ind_5/output/BiasAdd�
gen_190_ind_5/output/SoftmaxSoftmax%gen_190_ind_5/output/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
gen_190_ind_5/output/Softmax�
IdentityIdentity&gen_190_ind_5/output/Softmax:softmax:0,^gen_190_ind_5/conv_2/BiasAdd/ReadVariableOp+^gen_190_ind_5/conv_2/Conv2D/ReadVariableOp,^gen_190_ind_5/conv_3/BiasAdd/ReadVariableOp+^gen_190_ind_5/conv_3/Conv2D/ReadVariableOp,^gen_190_ind_5/dense1/BiasAdd/ReadVariableOp+^gen_190_ind_5/dense1/MatMul/ReadVariableOp+^gen_190_ind_5/input/BiasAdd/ReadVariableOp*^gen_190_ind_5/input/Conv2D/ReadVariableOp,^gen_190_ind_5/output/BiasAdd/ReadVariableOp+^gen_190_ind_5/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2Z
+gen_190_ind_5/conv_2/BiasAdd/ReadVariableOp+gen_190_ind_5/conv_2/BiasAdd/ReadVariableOp2X
*gen_190_ind_5/conv_2/Conv2D/ReadVariableOp*gen_190_ind_5/conv_2/Conv2D/ReadVariableOp2Z
+gen_190_ind_5/conv_3/BiasAdd/ReadVariableOp+gen_190_ind_5/conv_3/BiasAdd/ReadVariableOp2X
*gen_190_ind_5/conv_3/Conv2D/ReadVariableOp*gen_190_ind_5/conv_3/Conv2D/ReadVariableOp2Z
+gen_190_ind_5/dense1/BiasAdd/ReadVariableOp+gen_190_ind_5/dense1/BiasAdd/ReadVariableOp2X
*gen_190_ind_5/dense1/MatMul/ReadVariableOp*gen_190_ind_5/dense1/MatMul/ReadVariableOp2X
*gen_190_ind_5/input/BiasAdd/ReadVariableOp*gen_190_ind_5/input/BiasAdd/ReadVariableOp2V
)gen_190_ind_5/input/Conv2D/ReadVariableOp)gen_190_ind_5/input/Conv2D/ReadVariableOp2Z
+gen_190_ind_5/output/BiasAdd/ReadVariableOp+gen_190_ind_5/output/BiasAdd/ReadVariableOp2X
*gen_190_ind_5/output/MatMul/ReadVariableOp*gen_190_ind_5/output/MatMul/ReadVariableOp:+ '
%
_user_specified_nameinput_input
�
�
)__inference_conv_2_layer_call_fn_18289076

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_182890682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
^
B__inference_flat_layer_call_and_return_conditional_losses_18289396

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:& "
 
_user_specified_nameinputs
�
�
C__inference_input_layer_call_and_return_conditional_losses_18289047

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
D__inference_output_layer_call_and_return_conditional_losses_18289158

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
D__inference_dense1_layer_call_and_return_conditional_losses_18289412

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Sigmoid�
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
0__inference_gen_190_ind_5_layer_call_fn_18289390

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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_182892492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
D__inference_conv_3_layer_call_and_return_conditional_losses_18289089

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
^
B__inference_flat_layer_call_and_return_conditional_losses_18289116

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:& "
 
_user_specified_nameinputs
�1
�
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289319

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
identity��conv_2/BiasAdd/ReadVariableOp�conv_2/Conv2D/ReadVariableOp�conv_3/BiasAdd/ReadVariableOp�conv_3/Conv2D/ReadVariableOp�dense1/BiasAdd/ReadVariableOp�dense1/MatMul/ReadVariableOp�input/BiasAdd/ReadVariableOp�input/Conv2D/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
input/Conv2D/ReadVariableOpReadVariableOp$input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
input/Conv2D/ReadVariableOp�
input/Conv2DConv2Dinputs#input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
2
input/Conv2D�
input/BiasAdd/ReadVariableOpReadVariableOp%input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
input/BiasAdd/ReadVariableOp�
input/BiasAddBiasAddinput/Conv2D:output:0$input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
input/BiasAddt

input/ReluReluinput/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2

input/Relu�
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_2/Conv2D/ReadVariableOp�
conv_2/Conv2DConv2Dinput/Relu:activations:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������* *
paddingVALID*
strides
2
conv_2/Conv2D�
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_2/BiasAdd/ReadVariableOp�
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������* 2
conv_2/BiasAddu
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������* 2
conv_2/Relu�
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_3/Conv2D/ReadVariableOp�
conv_3/Conv2DConv2Dconv_2/Relu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
conv_3/Conv2D�
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOp�
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
2
conv_3/BiasAddu
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������
2
conv_3/Relui

flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2

flat/Const�
flat/ReshapeReshapeconv_3/Relu:activations:0flat/Const:output:0*
T0*(
_output_shapes
:����������2
flat/Reshape�
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
dense1/MatMul/ReadVariableOp�
dense1/MatMulMatMulflat/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense1/MatMul�
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense1/BiasAdd/ReadVariableOp�
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense1/BiasAddv
dense1/SigmoidSigmoiddense1/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense1/Sigmoid�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldense1/Sigmoid:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/MatMul�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
output/Softmax�
IdentityIdentityoutput/Softmax:softmax:0^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^input/BiasAdd/ReadVariableOp^input/Conv2D/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2>
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
�
�
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289191
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
identity��conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�dense1/StatefulPartitionedCall�input/StatefulPartitionedCall�output/StatefulPartitionedCall�
input/StatefulPartitionedCallStatefulPartitionedCallinput_input$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_182890472
input/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������* **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_182890682 
conv_2/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_182890892 
conv_3/StatefulPartitionedCall�
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_182891162
flat/PartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_182891352 
dense1/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_182891582 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
�
�
0__inference_gen_190_ind_5_layer_call_fn_18289375

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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_182892142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
(__inference_input_layer_call_fn_18289055

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_182890472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
)__inference_output_layer_call_fn_18289437

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_182891582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289249

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
identity��conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�dense1/StatefulPartitionedCall�input/StatefulPartitionedCall�output/StatefulPartitionedCall�
input/StatefulPartitionedCallStatefulPartitionedCallinputs$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:�����������**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_182890472
input/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������* **
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_182890682 
conv_2/StatefulPartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_182890892 
conv_3/StatefulPartitionedCall�
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_182891162
flat/PartitionedCall�
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_182891352 
dense1/StatefulPartitionedCall�
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_182891582 
output/StatefulPartitionedCall�
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
C
'__inference_flat_layer_call_fn_18289401

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_182891162
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
:& "
 
_user_specified_nameinputs
�1
�
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289360

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
identity��conv_2/BiasAdd/ReadVariableOp�conv_2/Conv2D/ReadVariableOp�conv_3/BiasAdd/ReadVariableOp�conv_3/Conv2D/ReadVariableOp�dense1/BiasAdd/ReadVariableOp�dense1/MatMul/ReadVariableOp�input/BiasAdd/ReadVariableOp�input/Conv2D/ReadVariableOp�output/BiasAdd/ReadVariableOp�output/MatMul/ReadVariableOp�
input/Conv2D/ReadVariableOpReadVariableOp$input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
input/Conv2D/ReadVariableOp�
input/Conv2DConv2Dinputs#input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������*
paddingVALID*
strides
2
input/Conv2D�
input/BiasAdd/ReadVariableOpReadVariableOp%input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
input/BiasAdd/ReadVariableOp�
input/BiasAddBiasAddinput/Conv2D:output:0$input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������2
input/BiasAddt

input/ReluReluinput/BiasAdd:output:0*
T0*1
_output_shapes
:�����������2

input/Relu�
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_2/Conv2D/ReadVariableOp�
conv_2/Conv2DConv2Dinput/Relu:activations:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������* *
paddingVALID*
strides
2
conv_2/Conv2D�
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_2/BiasAdd/ReadVariableOp�
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������* 2
conv_2/BiasAddu
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������* 2
conv_2/Relu�
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_3/Conv2D/ReadVariableOp�
conv_3/Conv2DConv2Dconv_2/Relu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
conv_3/Conv2D�
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOp�
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
2
conv_3/BiasAddu
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*/
_output_shapes
:���������
2
conv_3/Relui

flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   2

flat/Const�
flat/ReshapeReshapeconv_3/Relu:activations:0flat/Const:output:0*
T0*(
_output_shapes
:����������2
flat/Reshape�
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
dense1/MatMul/ReadVariableOp�
dense1/MatMulMatMulflat/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense1/MatMul�
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense1/BiasAdd/ReadVariableOp�
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense1/BiasAddv
dense1/SigmoidSigmoiddense1/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense1/Sigmoid�
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
output/MatMul/ReadVariableOp�
output/MatMulMatMuldense1/Sigmoid:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/MatMul�
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp�
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
output/Softmax�
IdentityIdentityoutput/Softmax:softmax:0^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^input/BiasAdd/ReadVariableOp^input/Conv2D/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::2>
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
�
�
)__inference_dense1_layer_call_fn_18289419

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

GPU 

CPU2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_182891352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�/
�
$__inference__traced_restore_18289533
file_prefix%
!assignvariableop_input_602_kernel%
!assignvariableop_1_input_602_bias(
$assignvariableop_2_conv_2_602_kernel&
"assignvariableop_3_conv_2_602_bias(
$assignvariableop_4_conv_3_602_kernel&
"assignvariableop_5_conv_3_602_bias(
$assignvariableop_6_dense1_602_kernel&
"assignvariableop_7_dense1_602_bias(
$assignvariableop_8_output_602_kernel&
"assignvariableop_9_output_602_bias
identity_11��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*�
value�B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
RestoreV2/shape_and_slices�
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

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_input_602_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_input_602_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv_2_602_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv_2_602_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv_3_602_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv_3_602_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp$assignvariableop_6_dense1_602_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense1_602_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp$assignvariableop_8_output_602_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_output_602_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
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
NoOp�
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10�
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
�
�
&__inference_signature_wrapper_18289278
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
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������**
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference__wrapped_model_182890342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:�����������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
�
�
D__inference_conv_2_layer_call_and_return_conditional_losses_18289068

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�#
�
!__inference__traced_save_18289491
file_prefix/
+savev2_input_602_kernel_read_readvariableop-
)savev2_input_602_bias_read_readvariableop0
,savev2_conv_2_602_kernel_read_readvariableop.
*savev2_conv_2_602_bias_read_readvariableop0
,savev2_conv_3_602_kernel_read_readvariableop.
*savev2_conv_3_602_bias_read_readvariableop0
,savev2_dense1_602_kernel_read_readvariableop.
*savev2_dense1_602_bias_read_readvariableop0
,savev2_output_602_kernel_read_readvariableop.
*savev2_output_602_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f127610672fa49c99a63bf0c858ceabc/part2
StringJoin/inputs_1�

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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*�
value�B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_input_602_kernel_read_readvariableop)savev2_input_602_bias_read_readvariableop,savev2_conv_2_602_kernel_read_readvariableop*savev2_conv_2_602_bias_read_readvariableop,savev2_conv_3_602_kernel_read_readvariableop*savev2_conv_3_602_bias_read_readvariableop,savev2_dense1_602_kernel_read_readvariableop*savev2_dense1_602_bias_read_readvariableop,savev2_output_602_kernel_read_readvariableop*savev2_output_602_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2
2
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapeso
m: :::::::	�
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
input_input>
serving_default_input_input:0�����������:
output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:ۿ
�0
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
regularization_losses
		variables

trainable_variables
	keras_api

signatures
K__call__
*L&call_and_return_all_conditional_losses
M_default_save_signature"�-
_tf_keras_sequential�-{"class_name": "Sequential", "name": "gen_190_ind_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gen_190_ind_5", "layers": [{"class_name": "Conv2D", "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "filters": 6, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [5, 5], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "gen_190_ind_5", "layers": [{"class_name": "Conv2D", "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "filters": 6, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [5, 5], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 210, 160, 4], "config": {"batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_input"}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
N__call__
*O&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "input", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 210, 160, 4], "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "filters": 6, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [5, 5], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 6}}}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
�
regularization_losses
 	variables
!trainable_variables
"	keras_api
T__call__
*U&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
V__call__
*W&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 140}}}}
�

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}}
 "
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
�
/non_trainable_variables

0layers
1layer_regularization_losses
regularization_losses
2metrics
		variables

trainable_variables
K__call__
M_default_save_signature
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
,
Zserving_default"
signature_map
*:(2input_602/kernel
:2input_602/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
3non_trainable_variables

4layers
5layer_regularization_losses
regularization_losses
6metrics
	variables
trainable_variables
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
+:)2conv_2_602/kernel
:2conv_2_602/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
7non_trainable_variables

8layers
9layer_regularization_losses
regularization_losses
:metrics
	variables
trainable_variables
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
+:)2conv_3_602/kernel
:2conv_3_602/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
;non_trainable_variables

<layers
=layer_regularization_losses
regularization_losses
>metrics
	variables
trainable_variables
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
�
?non_trainable_variables

@layers
Alayer_regularization_losses
regularization_losses
Bmetrics
 	variables
!trainable_variables
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
$:"	�
2dense1_602/kernel
:
2dense1_602/bias
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Elayer_regularization_losses
%regularization_losses
Fmetrics
&	variables
'trainable_variables
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
#:!
2output_602/kernel
:2output_602/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
Ilayer_regularization_losses
+regularization_losses
Jmetrics
,	variables
-trainable_variables
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
�2�
0__inference_gen_190_ind_5_layer_call_fn_18289375
0__inference_gen_190_ind_5_layer_call_fn_18289227
0__inference_gen_190_ind_5_layer_call_fn_18289390
0__inference_gen_190_ind_5_layer_call_fn_18289262�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289360
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289171
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289319
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289191�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference__wrapped_model_18289034�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *4�1
/�,
input_input�����������
�2�
(__inference_input_layer_call_fn_18289055�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
C__inference_input_layer_call_and_return_conditional_losses_18289047�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
)__inference_conv_2_layer_call_fn_18289076�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
D__inference_conv_2_layer_call_and_return_conditional_losses_18289068�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
)__inference_conv_3_layer_call_fn_18289097�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
D__inference_conv_3_layer_call_and_return_conditional_losses_18289089�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
'__inference_flat_layer_call_fn_18289401�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_flat_layer_call_and_return_conditional_losses_18289396�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense1_layer_call_fn_18289419�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense1_layer_call_and_return_conditional_losses_18289412�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_output_layer_call_fn_18289437�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_output_layer_call_and_return_conditional_losses_18289430�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
9B7
&__inference_signature_wrapper_18289278input_input�
#__inference__wrapped_model_18289034}
#$)*>�;
4�1
/�,
input_input�����������
� "/�,
*
output �
output����������
D__inference_conv_2_layer_call_and_return_conditional_losses_18289068�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
)__inference_conv_2_layer_call_fn_18289076�I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
D__inference_conv_3_layer_call_and_return_conditional_losses_18289089�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
)__inference_conv_3_layer_call_fn_18289097�I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
D__inference_dense1_layer_call_and_return_conditional_losses_18289412]#$0�-
&�#
!�
inputs����������
� "%�"
�
0���������

� }
)__inference_dense1_layer_call_fn_18289419P#$0�-
&�#
!�
inputs����������
� "����������
�
B__inference_flat_layer_call_and_return_conditional_losses_18289396a7�4
-�*
(�%
inputs���������

� "&�#
�
0����������
� 
'__inference_flat_layer_call_fn_18289401T7�4
-�*
(�%
inputs���������

� "������������
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289171{
#$)*F�C
<�9
/�,
input_input�����������
p

 
� "%�"
�
0���������
� �
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289191{
#$)*F�C
<�9
/�,
input_input�����������
p 

 
� "%�"
�
0���������
� �
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289319v
#$)*A�>
7�4
*�'
inputs�����������
p

 
� "%�"
�
0���������
� �
K__inference_gen_190_ind_5_layer_call_and_return_conditional_losses_18289360v
#$)*A�>
7�4
*�'
inputs�����������
p 

 
� "%�"
�
0���������
� �
0__inference_gen_190_ind_5_layer_call_fn_18289227n
#$)*F�C
<�9
/�,
input_input�����������
p

 
� "�����������
0__inference_gen_190_ind_5_layer_call_fn_18289262n
#$)*F�C
<�9
/�,
input_input�����������
p 

 
� "�����������
0__inference_gen_190_ind_5_layer_call_fn_18289375i
#$)*A�>
7�4
*�'
inputs�����������
p

 
� "�����������
0__inference_gen_190_ind_5_layer_call_fn_18289390i
#$)*A�>
7�4
*�'
inputs�����������
p 

 
� "�����������
C__inference_input_layer_call_and_return_conditional_losses_18289047�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
(__inference_input_layer_call_fn_18289055�I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
D__inference_output_layer_call_and_return_conditional_losses_18289430\)*/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� |
)__inference_output_layer_call_fn_18289437O)*/�,
%�"
 �
inputs���������

� "�����������
&__inference_signature_wrapper_18289278�
#$)*M�J
� 
C�@
>
input_input/�,
input_input�����������"/�,
*
output �
output���������