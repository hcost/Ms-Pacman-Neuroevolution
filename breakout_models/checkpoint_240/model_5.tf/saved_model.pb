╢░
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
shapeshapeИ"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108ощ
Ж
input_1049/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameinput_1049/kernel

%input_1049/kernel/Read/ReadVariableOpReadVariableOpinput_1049/kernel*&
_output_shapes
:*
dtype0
v
input_1049/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameinput_1049/bias
o
#input_1049/bias/Read/ReadVariableOpReadVariableOpinput_1049/bias*
_output_shapes
:*
dtype0
И
conv_2_1049/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv_2_1049/kernel
Б
&conv_2_1049/kernel/Read/ReadVariableOpReadVariableOpconv_2_1049/kernel*&
_output_shapes
:*
dtype0
x
conv_2_1049/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_2_1049/bias
q
$conv_2_1049/bias/Read/ReadVariableOpReadVariableOpconv_2_1049/bias*
_output_shapes
:*
dtype0
И
conv_3_1049/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv_3_1049/kernel
Б
&conv_3_1049/kernel/Read/ReadVariableOpReadVariableOpconv_3_1049/kernel*&
_output_shapes
:*
dtype0
x
conv_3_1049/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv_3_1049/bias
q
$conv_3_1049/bias/Read/ReadVariableOpReadVariableOpconv_3_1049/bias*
_output_shapes
:*
dtype0
Б
dense1_1049/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	М
*#
shared_namedense1_1049/kernel
z
&dense1_1049/kernel/Read/ReadVariableOpReadVariableOpdense1_1049/kernel*
_output_shapes
:	М
*
dtype0
x
dense1_1049/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense1_1049/bias
q
$dense1_1049/bias/Read/ReadVariableOpReadVariableOpdense1_1049/bias*
_output_shapes
:
*
dtype0
А
output_1049/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*#
shared_nameoutput_1049/kernel
y
&output_1049/kernel/Read/ReadVariableOpReadVariableOpoutput_1049/kernel*
_output_shapes

:
*
dtype0
x
output_1049/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameoutput_1049/bias
q
$output_1049/bias/Read/ReadVariableOpReadVariableOpoutput_1049/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Ю
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*┘
value╧B╠ B┼
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
Ъ
/non_trainable_variables

0layers
1layer_regularization_losses
regularization_losses
2metrics
		variables

trainable_variables
 
][
VARIABLE_VALUEinput_1049/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEinput_1049/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Ъ
3non_trainable_variables

4layers
5layer_regularization_losses
regularization_losses
6metrics
	variables
trainable_variables
^\
VARIABLE_VALUEconv_2_1049/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv_2_1049/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Ъ
7non_trainable_variables

8layers
9layer_regularization_losses
regularization_losses
:metrics
	variables
trainable_variables
^\
VARIABLE_VALUEconv_3_1049/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv_3_1049/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Ъ
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
Ъ
?non_trainable_variables

@layers
Alayer_regularization_losses
regularization_losses
Bmetrics
 	variables
!trainable_variables
^\
VARIABLE_VALUEdense1_1049/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense1_1049/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
Ъ
Cnon_trainable_variables

Dlayers
Elayer_regularization_losses
%regularization_losses
Fmetrics
&	variables
'trainable_variables
^\
VARIABLE_VALUEoutput_1049/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEoutput_1049/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
Ъ
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
Т
serving_default_input_inputPlaceholder*1
_output_shapes
:         ╥а*
dtype0*&
shape:         ╥а
ц
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_inputinput_1049/kernelinput_1049/biasconv_2_1049/kernelconv_2_1049/biasconv_3_1049/kernelconv_3_1049/biasdense1_1049/kerneldense1_1049/biasoutput_1049/kerneloutput_1049/bias*
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
CPU2J 8*/
f*R(
&__inference_signature_wrapper_32837064
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%input_1049/kernel/Read/ReadVariableOp#input_1049/bias/Read/ReadVariableOp&conv_2_1049/kernel/Read/ReadVariableOp$conv_2_1049/bias/Read/ReadVariableOp&conv_3_1049/kernel/Read/ReadVariableOp$conv_3_1049/bias/Read/ReadVariableOp&dense1_1049/kernel/Read/ReadVariableOp$dense1_1049/bias/Read/ReadVariableOp&output_1049/kernel/Read/ReadVariableOp$output_1049/bias/Read/ReadVariableOpConst*
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
!__inference__traced_save_32837277
╚
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_1049/kernelinput_1049/biasconv_2_1049/kernelconv_2_1049/biasconv_3_1049/kernelconv_3_1049/biasdense1_1049/kerneldense1_1049/biasoutput_1049/kerneloutput_1049/bias*
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
$__inference__traced_restore_32837319 л
ў
┐
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32836977
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
identityИвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвdense1/StatefulPartitionedCallвinput/StatefulPartitionedCallвoutput/StatefulPartitionedCallм
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
CPU2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_328368332
input/StatefulPartitionedCall╩
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
CPU2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_328368542 
conv_2/StatefulPartitionedCall╦
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
CPU2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_328368752 
conv_3/StatefulPartitionedCall╓
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
CPU2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_328369022
flat/PartitionedCall╣
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
CPU2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_328369212 
dense1/StatefulPartitionedCall├
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
CPU2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_328369442 
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
└
к
)__inference_conv_3_layer_call_fn_32836883

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallа
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
CPU2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_328368752
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
▒
═
&__inference_signature_wrapper_32837064
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
identityИвStatefulPartitionedCallє
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
CPU2J 8*,
f'R%
#__inference__wrapped_model_328368202
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
ў
┐
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32836957
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
identityИвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвdense1/StatefulPartitionedCallвinput/StatefulPartitionedCallвoutput/StatefulPartitionedCallм
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
CPU2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_328368332
input/StatefulPartitionedCall╩
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
CPU2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_328368542 
conv_2/StatefulPartitionedCall╦
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
CPU2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_328368752 
conv_3/StatefulPartitionedCall╓
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
CPU2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_328369022
flat/PartitionedCall╣
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
CPU2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_328369212 
dense1/StatefulPartitionedCall├
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
CPU2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_328369442 
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
╛
й
(__inference_input_layer_call_fn_32836841

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
-:+                           **
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_328368332
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
╜@
╔
#__inference__wrapped_model_32836820
input_input6
2gen_240_ind_2_input_conv2d_readvariableop_resource7
3gen_240_ind_2_input_biasadd_readvariableop_resource7
3gen_240_ind_2_conv_2_conv2d_readvariableop_resource8
4gen_240_ind_2_conv_2_biasadd_readvariableop_resource7
3gen_240_ind_2_conv_3_conv2d_readvariableop_resource8
4gen_240_ind_2_conv_3_biasadd_readvariableop_resource7
3gen_240_ind_2_dense1_matmul_readvariableop_resource8
4gen_240_ind_2_dense1_biasadd_readvariableop_resource7
3gen_240_ind_2_output_matmul_readvariableop_resource8
4gen_240_ind_2_output_biasadd_readvariableop_resource
identityИв+gen_240_ind_2/conv_2/BiasAdd/ReadVariableOpв*gen_240_ind_2/conv_2/Conv2D/ReadVariableOpв+gen_240_ind_2/conv_3/BiasAdd/ReadVariableOpв*gen_240_ind_2/conv_3/Conv2D/ReadVariableOpв+gen_240_ind_2/dense1/BiasAdd/ReadVariableOpв*gen_240_ind_2/dense1/MatMul/ReadVariableOpв*gen_240_ind_2/input/BiasAdd/ReadVariableOpв)gen_240_ind_2/input/Conv2D/ReadVariableOpв+gen_240_ind_2/output/BiasAdd/ReadVariableOpв*gen_240_ind_2/output/MatMul/ReadVariableOp╤
)gen_240_ind_2/input/Conv2D/ReadVariableOpReadVariableOp2gen_240_ind_2_input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)gen_240_ind_2/input/Conv2D/ReadVariableOpч
gen_240_ind_2/input/Conv2DConv2Dinput_input1gen_240_ind_2/input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╨Ю*
paddingVALID*
strides
2
gen_240_ind_2/input/Conv2D╚
*gen_240_ind_2/input/BiasAdd/ReadVariableOpReadVariableOp3gen_240_ind_2_input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*gen_240_ind_2/input/BiasAdd/ReadVariableOp┌
gen_240_ind_2/input/BiasAddBiasAdd#gen_240_ind_2/input/Conv2D:output:02gen_240_ind_2/input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ╨Ю2
gen_240_ind_2/input/BiasAddЮ
gen_240_ind_2/input/ReluRelu$gen_240_ind_2/input/BiasAdd:output:0*
T0*1
_output_shapes
:         ╨Ю2
gen_240_ind_2/input/Relu╘
*gen_240_ind_2/conv_2/Conv2D/ReadVariableOpReadVariableOp3gen_240_ind_2_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*gen_240_ind_2/conv_2/Conv2D/ReadVariableOpГ
gen_240_ind_2/conv_2/Conv2DConv2D&gen_240_ind_2/input/Relu:activations:02gen_240_ind_2/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         * *
paddingVALID*
strides
2
gen_240_ind_2/conv_2/Conv2D╦
+gen_240_ind_2/conv_2/BiasAdd/ReadVariableOpReadVariableOp4gen_240_ind_2_conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+gen_240_ind_2/conv_2/BiasAdd/ReadVariableOp▄
gen_240_ind_2/conv_2/BiasAddBiasAdd$gen_240_ind_2/conv_2/Conv2D:output:03gen_240_ind_2/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         * 2
gen_240_ind_2/conv_2/BiasAddЯ
gen_240_ind_2/conv_2/ReluRelu%gen_240_ind_2/conv_2/BiasAdd:output:0*
T0*/
_output_shapes
:         * 2
gen_240_ind_2/conv_2/Relu╘
*gen_240_ind_2/conv_3/Conv2D/ReadVariableOpReadVariableOp3gen_240_ind_2_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*gen_240_ind_2/conv_3/Conv2D/ReadVariableOpД
gen_240_ind_2/conv_3/Conv2DConv2D'gen_240_ind_2/conv_2/Relu:activations:02gen_240_ind_2/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
gen_240_ind_2/conv_3/Conv2D╦
+gen_240_ind_2/conv_3/BiasAdd/ReadVariableOpReadVariableOp4gen_240_ind_2_conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+gen_240_ind_2/conv_3/BiasAdd/ReadVariableOp▄
gen_240_ind_2/conv_3/BiasAddBiasAdd$gen_240_ind_2/conv_3/Conv2D:output:03gen_240_ind_2/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
gen_240_ind_2/conv_3/BiasAddЯ
gen_240_ind_2/conv_3/ReluRelu%gen_240_ind_2/conv_3/BiasAdd:output:0*
T0*/
_output_shapes
:         
2
gen_240_ind_2/conv_3/ReluЕ
gen_240_ind_2/flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"    М   2
gen_240_ind_2/flat/Const┬
gen_240_ind_2/flat/ReshapeReshape'gen_240_ind_2/conv_3/Relu:activations:0!gen_240_ind_2/flat/Const:output:0*
T0*(
_output_shapes
:         М2
gen_240_ind_2/flat/Reshape═
*gen_240_ind_2/dense1/MatMul/ReadVariableOpReadVariableOp3gen_240_ind_2_dense1_matmul_readvariableop_resource*
_output_shapes
:	М
*
dtype02,
*gen_240_ind_2/dense1/MatMul/ReadVariableOp╧
gen_240_ind_2/dense1/MatMulMatMul#gen_240_ind_2/flat/Reshape:output:02gen_240_ind_2/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
gen_240_ind_2/dense1/MatMul╦
+gen_240_ind_2/dense1/BiasAdd/ReadVariableOpReadVariableOp4gen_240_ind_2_dense1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+gen_240_ind_2/dense1/BiasAdd/ReadVariableOp╒
gen_240_ind_2/dense1/BiasAddBiasAdd%gen_240_ind_2/dense1/MatMul:product:03gen_240_ind_2/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
gen_240_ind_2/dense1/BiasAddа
gen_240_ind_2/dense1/SigmoidSigmoid%gen_240_ind_2/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
gen_240_ind_2/dense1/Sigmoid╠
*gen_240_ind_2/output/MatMul/ReadVariableOpReadVariableOp3gen_240_ind_2_output_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*gen_240_ind_2/output/MatMul/ReadVariableOp╠
gen_240_ind_2/output/MatMulMatMul gen_240_ind_2/dense1/Sigmoid:y:02gen_240_ind_2/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
gen_240_ind_2/output/MatMul╦
+gen_240_ind_2/output/BiasAdd/ReadVariableOpReadVariableOp4gen_240_ind_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+gen_240_ind_2/output/BiasAdd/ReadVariableOp╒
gen_240_ind_2/output/BiasAddBiasAdd%gen_240_ind_2/output/MatMul:product:03gen_240_ind_2/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
gen_240_ind_2/output/BiasAddа
gen_240_ind_2/output/SoftmaxSoftmax%gen_240_ind_2/output/BiasAdd:output:0*
T0*'
_output_shapes
:         2
gen_240_ind_2/output/Softmax┐
IdentityIdentity&gen_240_ind_2/output/Softmax:softmax:0,^gen_240_ind_2/conv_2/BiasAdd/ReadVariableOp+^gen_240_ind_2/conv_2/Conv2D/ReadVariableOp,^gen_240_ind_2/conv_3/BiasAdd/ReadVariableOp+^gen_240_ind_2/conv_3/Conv2D/ReadVariableOp,^gen_240_ind_2/dense1/BiasAdd/ReadVariableOp+^gen_240_ind_2/dense1/MatMul/ReadVariableOp+^gen_240_ind_2/input/BiasAdd/ReadVariableOp*^gen_240_ind_2/input/Conv2D/ReadVariableOp,^gen_240_ind_2/output/BiasAdd/ReadVariableOp+^gen_240_ind_2/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*X
_input_shapesG
E:         ╥а::::::::::2Z
+gen_240_ind_2/conv_2/BiasAdd/ReadVariableOp+gen_240_ind_2/conv_2/BiasAdd/ReadVariableOp2X
*gen_240_ind_2/conv_2/Conv2D/ReadVariableOp*gen_240_ind_2/conv_2/Conv2D/ReadVariableOp2Z
+gen_240_ind_2/conv_3/BiasAdd/ReadVariableOp+gen_240_ind_2/conv_3/BiasAdd/ReadVariableOp2X
*gen_240_ind_2/conv_3/Conv2D/ReadVariableOp*gen_240_ind_2/conv_3/Conv2D/ReadVariableOp2Z
+gen_240_ind_2/dense1/BiasAdd/ReadVariableOp+gen_240_ind_2/dense1/BiasAdd/ReadVariableOp2X
*gen_240_ind_2/dense1/MatMul/ReadVariableOp*gen_240_ind_2/dense1/MatMul/ReadVariableOp2X
*gen_240_ind_2/input/BiasAdd/ReadVariableOp*gen_240_ind_2/input/BiasAdd/ReadVariableOp2V
)gen_240_ind_2/input/Conv2D/ReadVariableOp)gen_240_ind_2/input/Conv2D/ReadVariableOp2Z
+gen_240_ind_2/output/BiasAdd/ReadVariableOp+gen_240_ind_2/output/BiasAdd/ReadVariableOp2X
*gen_240_ind_2/output/MatMul/ReadVariableOp*gen_240_ind_2/output/MatMul/ReadVariableOp:+ '
%
_user_specified_nameinput_input
└
к
)__inference_conv_2_layer_call_fn_32836862

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallа
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
CPU2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_328368542
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
у
╫
0__inference_gen_240_ind_2_layer_call_fn_32837013
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
identityИвStatefulPartitionedCallЫ
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
CPU2J 8*T
fORM
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_328370002
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
ш
▄
C__inference_input_layer_call_and_return_conditional_losses_32836833

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
ч#
Ў
!__inference__traced_save_32837277
file_prefix0
,savev2_input_1049_kernel_read_readvariableop.
*savev2_input_1049_bias_read_readvariableop1
-savev2_conv_2_1049_kernel_read_readvariableop/
+savev2_conv_2_1049_bias_read_readvariableop1
-savev2_conv_3_1049_kernel_read_readvariableop/
+savev2_conv_3_1049_bias_read_readvariableop1
-savev2_dense1_1049_kernel_read_readvariableop/
+savev2_dense1_1049_bias_read_readvariableop1
-savev2_output_1049_kernel_read_readvariableop/
+savev2_output_1049_bias_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1е
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_3bded49cde3e4ba5a058dd672118268d/part2
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
SaveV2/shape_and_slices 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_input_1049_kernel_read_readvariableop*savev2_input_1049_bias_read_readvariableop-savev2_conv_2_1049_kernel_read_readvariableop+savev2_conv_2_1049_bias_read_readvariableop-savev2_conv_3_1049_kernel_read_readvariableop+savev2_conv_3_1049_bias_read_readvariableop-savev2_dense1_1049_kernel_read_readvariableop+savev2_dense1_1049_bias_read_readvariableop-savev2_output_1049_kernel_read_readvariableop+savev2_output_1049_bias_read_readvariableop"/device:CPU:0*
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
у
╫
0__inference_gen_240_ind_2_layer_call_fn_32837048
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
identityИвStatefulPartitionedCallЫ
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
CPU2J 8*T
fORM
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_328370352
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
ш
║
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32837035

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
identityИвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвdense1/StatefulPartitionedCallвinput/StatefulPartitionedCallвoutput/StatefulPartitionedCallз
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
CPU2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_328368332
input/StatefulPartitionedCall╩
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
CPU2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_328368542 
conv_2/StatefulPartitionedCall╦
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
CPU2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_328368752 
conv_3/StatefulPartitionedCall╓
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
CPU2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_328369022
flat/PartitionedCall╣
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
CPU2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_328369212 
dense1/StatefulPartitionedCall├
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
CPU2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_328369442 
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
═	
▌
D__inference_output_layer_call_and_return_conditional_losses_32837216

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
щ
▌
D__inference_conv_3_layer_call_and_return_conditional_losses_32836875

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
╔	
▌
D__inference_dense1_layer_call_and_return_conditional_losses_32836921

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
═	
▌
D__inference_output_layer_call_and_return_conditional_losses_32836944

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
ш
║
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32837000

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
identityИвconv_2/StatefulPartitionedCallвconv_3/StatefulPartitionedCallвdense1/StatefulPartitionedCallвinput/StatefulPartitionedCallвoutput/StatefulPartitionedCallз
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
CPU2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_328368332
input/StatefulPartitionedCall╩
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
CPU2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_328368542 
conv_2/StatefulPartitionedCall╦
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
CPU2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_328368752 
conv_3/StatefulPartitionedCall╓
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
CPU2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_328369022
flat/PartitionedCall╣
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
CPU2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_328369212 
dense1/StatefulPartitionedCall├
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
CPU2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_328369442 
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
 /
╒
$__inference__traced_restore_32837319
file_prefix&
"assignvariableop_input_1049_kernel&
"assignvariableop_1_input_1049_bias)
%assignvariableop_2_conv_2_1049_kernel'
#assignvariableop_3_conv_2_1049_bias)
%assignvariableop_4_conv_3_1049_kernel'
#assignvariableop_5_conv_3_1049_bias)
%assignvariableop_6_dense1_1049_kernel'
#assignvariableop_7_dense1_1049_bias)
%assignvariableop_8_output_1049_kernel'
#assignvariableop_9_output_1049_bias
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

IdentityТ
AssignVariableOpAssignVariableOp"assignvariableop_input_1049_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ш
AssignVariableOp_1AssignVariableOp"assignvariableop_1_input_1049_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ы
AssignVariableOp_2AssignVariableOp%assignvariableop_2_conv_2_1049_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Щ
AssignVariableOp_3AssignVariableOp#assignvariableop_3_conv_2_1049_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ы
AssignVariableOp_4AssignVariableOp%assignvariableop_4_conv_3_1049_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Щ
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv_3_1049_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ы
AssignVariableOp_6AssignVariableOp%assignvariableop_6_dense1_1049_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Щ
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense1_1049_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ы
AssignVariableOp_8AssignVariableOp%assignvariableop_8_output_1049_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Щ
AssignVariableOp_9AssignVariableOp#assignvariableop_9_output_1049_biasIdentity_9:output:0*
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
Ъ1
╘
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32837105

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
Й
^
B__inference_flat_layer_call_and_return_conditional_losses_32837182

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
Ъ1
╘
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32837146

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
╔	
▌
D__inference_dense1_layer_call_and_return_conditional_losses_32837198

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
╘
╥
0__inference_gen_240_ind_2_layer_call_fn_32837161

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
identityИвStatefulPartitionedCallЦ
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
CPU2J 8*T
fORM
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_328370002
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
Є
к
)__inference_output_layer_call_fn_32837223

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЖ
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
CPU2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_328369442
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
 
_user_specified_nameinputs
Й
^
B__inference_flat_layer_call_and_return_conditional_losses_32836902

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
┘
C
'__inference_flat_layer_call_fn_32837187

inputs
identityл
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
CPU2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_328369022
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
щ
▌
D__inference_conv_2_layer_call_and_return_conditional_losses_32836854

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
╘
╥
0__inference_gen_240_ind_2_layer_call_fn_32837176

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
identityИвStatefulPartitionedCallЦ
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
CPU2J 8*T
fORM
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_328370352
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
є
к
)__inference_dense1_layer_call_fn_32837205

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЖ
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
CPU2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_328369212
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
StatefulPartitionedCall:0         tensorflow/serving/predict:х┐
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
regularization_losses
		variables

trainable_variables
	keras_api

signatures
K__call__
*L&call_and_return_all_conditional_losses
M_default_save_signature"▀-
_tf_keras_sequential└-{"class_name": "Sequential", "name": "gen_240_ind_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gen_240_ind_2", "layers": [{"class_name": "Conv2D", "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "filters": 6, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [5, 5], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "gen_240_ind_2", "layers": [{"class_name": "Conv2D", "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "filters": 6, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [5, 5], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
╣"╢
_tf_keras_input_layerЦ{"class_name": "InputLayer", "name": "input_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 210, 160, 4], "config": {"batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_input"}}
Ю

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
N__call__
*O&call_and_return_all_conditional_losses"∙
_tf_keras_layer▀{"class_name": "Conv2D", "name": "input", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 210, 160, 4], "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 4], "dtype": "float32", "filters": 6, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}}
ч

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"┬
_tf_keras_layerи{"class_name": "Conv2D", "name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": [3, 3], "strides": [5, 5], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 6}}}}
ч

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R__call__
*S&call_and_return_all_conditional_losses"┬
_tf_keras_layerи{"class_name": "Conv2D", "name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [3, 3], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
ж
regularization_losses
 	variables
!trainable_variables
"	keras_api
T__call__
*U&call_and_return_all_conditional_losses"Ч
_tf_keras_layer¤{"class_name": "Flatten", "name": "flat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
є

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
V__call__
*W&call_and_return_all_conditional_losses"╬
_tf_keras_layer┤{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 10, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 140}}}}
ё

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"╠
_tf_keras_layer▓{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}}
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
╖
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
+:)2input_1049/kernel
:2input_1049/bias
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
Ъ
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
,:*2conv_2_1049/kernel
:2conv_2_1049/bias
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
Ъ
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
,:*2conv_3_1049/kernel
:2conv_3_1049/bias
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
Ъ
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
Ъ
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
%:#	М
2dense1_1049/kernel
:
2dense1_1049/bias
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
Ъ
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
$:"
2output_1049/kernel
:2output_1049/bias
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
Ъ
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
О2Л
0__inference_gen_240_ind_2_layer_call_fn_32837048
0__inference_gen_240_ind_2_layer_call_fn_32837161
0__inference_gen_240_ind_2_layer_call_fn_32837013
0__inference_gen_240_ind_2_layer_call_fn_32837176└
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
·2ў
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32836977
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32836957
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32837146
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32837105└
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
я2ь
#__inference__wrapped_model_32836820─
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
З2Д
(__inference_input_layer_call_fn_32836841╫
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
в2Я
C__inference_input_layer_call_and_return_conditional_losses_32836833╫
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
И2Е
)__inference_conv_2_layer_call_fn_32836862╫
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
г2а
D__inference_conv_2_layer_call_and_return_conditional_losses_32836854╫
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
И2Е
)__inference_conv_3_layer_call_fn_32836883╫
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
г2а
D__inference_conv_3_layer_call_and_return_conditional_losses_32836875╫
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
╤2╬
'__inference_flat_layer_call_fn_32837187в
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
ь2щ
B__inference_flat_layer_call_and_return_conditional_losses_32837182в
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
╙2╨
)__inference_dense1_layer_call_fn_32837205в
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
ю2ы
D__inference_dense1_layer_call_and_return_conditional_losses_32837198в
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
╙2╨
)__inference_output_layer_call_fn_32837223в
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
ю2ы
D__inference_output_layer_call_and_return_conditional_losses_32837216в
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
9B7
&__inference_signature_wrapper_32837064input_inputд
#__inference__wrapped_model_32836820}
#$)*>в;
4в1
/К,
input_input         ╥а
к "/к,
*
output К
output         ┘
D__inference_conv_2_layer_call_and_return_conditional_losses_32836854РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ▒
)__inference_conv_2_layer_call_fn_32836862ГIвF
?в<
:К7
inputs+                           
к "2К/+                           ┘
D__inference_conv_3_layer_call_and_return_conditional_losses_32836875РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ▒
)__inference_conv_3_layer_call_fn_32836883ГIвF
?в<
:К7
inputs+                           
к "2К/+                           е
D__inference_dense1_layer_call_and_return_conditional_losses_32837198]#$0в-
&в#
!К
inputs         М
к "%в"
К
0         

Ъ }
)__inference_dense1_layer_call_fn_32837205P#$0в-
&в#
!К
inputs         М
к "К         
з
B__inference_flat_layer_call_and_return_conditional_losses_32837182a7в4
-в*
(К%
inputs         

к "&в#
К
0         М
Ъ 
'__inference_flat_layer_call_fn_32837187T7в4
-в*
(К%
inputs         

к "К         М╩
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32836957{
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
Ъ ╩
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32836977{
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
Ъ ┼
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32837105v
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
Ъ ┼
K__inference_gen_240_ind_2_layer_call_and_return_conditional_losses_32837146v
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
Ъ в
0__inference_gen_240_ind_2_layer_call_fn_32837013n
#$)*FвC
<в9
/К,
input_input         ╥а
p

 
к "К         в
0__inference_gen_240_ind_2_layer_call_fn_32837048n
#$)*FвC
<в9
/К,
input_input         ╥а
p 

 
к "К         Э
0__inference_gen_240_ind_2_layer_call_fn_32837161i
#$)*Aв>
7в4
*К'
inputs         ╥а
p

 
к "К         Э
0__inference_gen_240_ind_2_layer_call_fn_32837176i
#$)*Aв>
7в4
*К'
inputs         ╥а
p 

 
к "К         ╪
C__inference_input_layer_call_and_return_conditional_losses_32836833РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ░
(__inference_input_layer_call_fn_32836841ГIвF
?в<
:К7
inputs+                           
к "2К/+                           д
D__inference_output_layer_call_and_return_conditional_losses_32837216\)*/в,
%в"
 К
inputs         

к "%в"
К
0         
Ъ |
)__inference_output_layer_call_fn_32837223O)*/в,
%в"
 К
inputs         

к "К         ╖
&__inference_signature_wrapper_32837064М
#$)*MвJ
в 
Cк@
>
input_input/К,
input_input         ╥а"/к,
*
output К
output         