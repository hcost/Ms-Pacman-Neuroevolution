аш	
Ћ§
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
dtypetype
О
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02v2.1.0-rc2-17-ge5bf8de4108ј

input_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameinput_87/kernel
{
#input_87/kernel/Read/ReadVariableOpReadVariableOpinput_87/kernel*&
_output_shapes
:*
dtype0
r
input_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameinput_87/bias
k
!input_87/bias/Read/ReadVariableOpReadVariableOpinput_87/bias*
_output_shapes
:*
dtype0
r
norm_87/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namenorm_87/gamma
k
!norm_87/gamma/Read/ReadVariableOpReadVariableOpnorm_87/gamma*
_output_shapes
:*
dtype0
p
norm_87/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namenorm_87/beta
i
 norm_87/beta/Read/ReadVariableOpReadVariableOpnorm_87/beta*
_output_shapes
:*
dtype0
~
norm_87/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namenorm_87/moving_mean
w
'norm_87/moving_mean/Read/ReadVariableOpReadVariableOpnorm_87/moving_mean*
_output_shapes
:*
dtype0

norm_87/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namenorm_87/moving_variance

+norm_87/moving_variance/Read/ReadVariableOpReadVariableOpnorm_87/moving_variance*
_output_shapes
:*
dtype0

conv_2_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_nameconv_2_87/kernel
}
$conv_2_87/kernel/Read/ReadVariableOpReadVariableOpconv_2_87/kernel*&
_output_shapes
:	*
dtype0
t
conv_2_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv_2_87/bias
m
"conv_2_87/bias/Read/ReadVariableOpReadVariableOpconv_2_87/bias*
_output_shapes
:	*
dtype0

conv_3_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_nameconv_3_87/kernel
}
$conv_3_87/kernel/Read/ReadVariableOpReadVariableOpconv_3_87/kernel*&
_output_shapes
:	*
dtype0
t
conv_3_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3_87/bias
m
"conv_3_87/bias/Read/ReadVariableOpReadVariableOpconv_3_87/bias*
_output_shapes
:*
dtype0
}
dense1_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	N*!
shared_namedense1_87/kernel
v
$dense1_87/kernel/Read/ReadVariableOpReadVariableOpdense1_87/kernel*
_output_shapes
:	N*
dtype0
t
dense1_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense1_87/bias
m
"dense1_87/bias/Read/ReadVariableOpReadVariableOpdense1_87/bias*
_output_shapes
:*
dtype0
|
output_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_nameoutput_87/kernel
u
$output_87/kernel/Read/ReadVariableOpReadVariableOpoutput_87/kernel*
_output_shapes

:*
dtype0
t
output_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput_87/bias
m
"output_87/bias/Read/ReadVariableOpReadVariableOpoutput_87/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ћ!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ж!
valueЌ!BЉ! BЂ!
ц
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		variables

regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api

axis
	gamma
beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
h

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
R
)	variables
*regularization_losses
+trainable_variables
,	keras_api
h

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
h

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
f
0
1
2
3
4
5
6
7
#8
$9
-10
.11
312
413
 
V
0
1
2
3
4
5
#6
$7
-8
.9
310
411

9metrics
		variables

regularization_losses
:non_trainable_variables
;layer_regularization_losses

<layers
trainable_variables
 
[Y
VARIABLE_VALUEinput_87/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEinput_87/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1

=metrics
	variables
regularization_losses
>non_trainable_variables
?layer_regularization_losses

@layers
trainable_variables
 
XV
VARIABLE_VALUEnorm_87/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEnorm_87/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEnorm_87/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEnorm_87/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 

0
1

Ametrics
	variables
regularization_losses
Bnon_trainable_variables
Clayer_regularization_losses

Dlayers
trainable_variables
\Z
VARIABLE_VALUEconv_2_87/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv_2_87/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1

Emetrics
	variables
 regularization_losses
Fnon_trainable_variables
Glayer_regularization_losses

Hlayers
!trainable_variables
\Z
VARIABLE_VALUEconv_3_87/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv_3_87/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1

Imetrics
%	variables
&regularization_losses
Jnon_trainable_variables
Klayer_regularization_losses

Llayers
'trainable_variables
 
 
 

Mmetrics
)	variables
*regularization_losses
Nnon_trainable_variables
Olayer_regularization_losses

Players
+trainable_variables
\Z
VARIABLE_VALUEdense1_87/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense1_87/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1
 

-0
.1

Qmetrics
/	variables
0regularization_losses
Rnon_trainable_variables
Slayer_regularization_losses

Tlayers
1trainable_variables
\Z
VARIABLE_VALUEoutput_87/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEoutput_87/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41

Umetrics
5	variables
6regularization_losses
Vnon_trainable_variables
Wlayer_regularization_losses

Xlayers
7trainable_variables
 

0
1
 
1
0
1
2
3
4
5
6
 
 
 
 
 

0
1
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

serving_default_input_inputPlaceholder*1
_output_shapes
:џџџџџџџџџв *
dtype0*&
shape:џџџџџџџџџв 
Ё
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_inputinput_87/kernelinput_87/biasnorm_87/gammanorm_87/betanorm_87/moving_meannorm_87/moving_varianceconv_2_87/kernelconv_2_87/biasconv_3_87/kernelconv_3_87/biasdense1_87/kerneldense1_87/biasoutput_87/kerneloutput_87/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*/
f*R(
&__inference_signature_wrapper_10873308
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#input_87/kernel/Read/ReadVariableOp!input_87/bias/Read/ReadVariableOp!norm_87/gamma/Read/ReadVariableOp norm_87/beta/Read/ReadVariableOp'norm_87/moving_mean/Read/ReadVariableOp+norm_87/moving_variance/Read/ReadVariableOp$conv_2_87/kernel/Read/ReadVariableOp"conv_2_87/bias/Read/ReadVariableOp$conv_3_87/kernel/Read/ReadVariableOp"conv_3_87/bias/Read/ReadVariableOp$dense1_87/kernel/Read/ReadVariableOp"dense1_87/bias/Read/ReadVariableOp$output_87/kernel/Read/ReadVariableOp"output_87/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__traced_save_10873749

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_87/kernelinput_87/biasnorm_87/gammanorm_87/betanorm_87/moving_meannorm_87/moving_varianceconv_2_87/kernelconv_2_87/biasconv_3_87/kernelconv_3_87/biasdense1_87/kerneldense1_87/biasoutput_87/kerneloutput_87/bias*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference__traced_restore_10873803Ћ
и>
Ф
$__inference__traced_restore_10873803
file_prefix$
 assignvariableop_input_87_kernel$
 assignvariableop_1_input_87_bias$
 assignvariableop_2_norm_87_gamma#
assignvariableop_3_norm_87_beta*
&assignvariableop_4_norm_87_moving_mean.
*assignvariableop_5_norm_87_moving_variance'
#assignvariableop_6_conv_2_87_kernel%
!assignvariableop_7_conv_2_87_bias'
#assignvariableop_8_conv_3_87_kernel%
!assignvariableop_9_conv_3_87_bias(
$assignvariableop_10_dense1_87_kernel&
"assignvariableop_11_dense1_87_bias(
$assignvariableop_12_output_87_kernel&
"assignvariableop_13_output_87_bias
identity_15ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*І
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesЊ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesё
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_input_87_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp assignvariableop_1_input_87_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOp assignvariableop_2_norm_87_gammaIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOpassignvariableop_3_norm_87_betaIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOp&assignvariableop_4_norm_87_moving_meanIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5 
AssignVariableOp_5AssignVariableOp*assignvariableop_5_norm_87_moving_varianceIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv_2_87_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv_2_87_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv_3_87_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv_3_87_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense1_87_kernelIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense1_87_biasIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12
AssignVariableOp_12AssignVariableOp$assignvariableop_12_output_87_kernelIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13
AssignVariableOp_13AssignVariableOp"assignvariableop_13_output_87_biasIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13Ј
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesФ
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
NoOp
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
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
ш
м
C__inference_input_layer_call_and_return_conditional_losses_10872827

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Ї&
ќ
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873215

inputs(
$input_statefulpartitionedcall_args_1(
$input_statefulpartitionedcall_args_2'
#norm_statefulpartitionedcall_args_1'
#norm_statefulpartitionedcall_args_2'
#norm_statefulpartitionedcall_args_3'
#norm_statefulpartitionedcall_args_4)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2)
%dense1_statefulpartitionedcall_args_1)
%dense1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identityЂconv_2/StatefulPartitionedCallЂconv_3/StatefulPartitionedCallЂdense1/StatefulPartitionedCallЂinput/StatefulPartitionedCallЂnorm/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЇ
input/StatefulPartitionedCallStatefulPartitionedCallinputs$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:џџџџџџџџџа**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_108728272
input/StatefulPartitionedCall
norm/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0#norm_statefulpartitionedcall_args_1#norm_statefulpartitionedcall_args_2#norm_statefulpartitionedcall_args_3#norm_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:џџџџџџџџџа**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_norm_layer_call_and_return_conditional_losses_108730492
norm/StatefulPartitionedCallЩ
conv_2/StatefulPartitionedCallStatefulPartitionedCall%norm/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџfM	**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_108729802 
conv_2/StatefulPartitionedCallЫ
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџ **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_108730012 
conv_3/StatefulPartitionedCallж
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:џџџџџџџџџN**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_108731072
flat/PartitionedCallЙ
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_108731262 
dense1/StatefulPartitionedCallУ
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_108731492 
output/StatefulPartitionedCallО
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^norm/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2<
norm/StatefulPartitionedCallnorm/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
щ
н
D__inference_conv_2_layer_call_and_return_conditional_losses_10872980

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs

ъ
/__inference_gen_60_ind_3_layer_call_fn_10873276
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
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinput_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_108732592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
Т
х
B__inference_norm_layer_call_and_return_conditional_losses_10873071

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџа:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
Constм
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:џџџџџџџџџа2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџа::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
й
C
'__inference_flat_layer_call_fn_10873647

inputs
identityЋ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:џџџџџџџџџN**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_108731072
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
Ы$

B__inference_norm_layer_call_and_return_conditional_losses_10872929

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_10872914
assignmovingavg_1_10872921
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2	
Const_2 
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/10872914*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xБ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10872914*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10872914*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpЮ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/10872914*
_output_shapes
:2
AssignMovingAvg/sub_1З
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/10872914*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10872914AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/10872914*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpІ
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/10872921*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЙ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10872921*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10872921*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10872921*
_output_shapes
:2
AssignMovingAvg_1/sub_1С
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10872921*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10872921AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/10872921*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Э	
н
D__inference_output_layer_call_and_return_conditional_losses_10873149

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Щ	
н
D__inference_dense1_layer_call_and_return_conditional_losses_10873126

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	N*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџN::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
О
Љ
(__inference_input_layer_call_fn_10872835

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_108728272
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ѓ
Њ
)__inference_dense1_layer_call_fn_10873665

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_108731262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџN::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ю
№
'__inference_norm_layer_call_fn_10873562

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_norm_layer_call_and_return_conditional_losses_108729602
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ђ
Њ
)__inference_output_layer_call_fn_10873683

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_108731492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

№
'__inference_norm_layer_call_fn_10873627

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:џџџџџџџџџа**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_norm_layer_call_and_return_conditional_losses_108730492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџа2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџа::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
W
Е	
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873379

inputs(
$input_conv2d_readvariableop_resource)
%input_biasadd_readvariableop_resource 
norm_readvariableop_resource"
norm_readvariableop_1_resource!
norm_assignmovingavg_10873334#
norm_assignmovingavg_1_10873341)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityЂconv_2/BiasAdd/ReadVariableOpЂconv_2/Conv2D/ReadVariableOpЂconv_3/BiasAdd/ReadVariableOpЂconv_3/Conv2D/ReadVariableOpЂdense1/BiasAdd/ReadVariableOpЂdense1/MatMul/ReadVariableOpЂinput/BiasAdd/ReadVariableOpЂinput/Conv2D/ReadVariableOpЂ(norm/AssignMovingAvg/AssignSubVariableOpЂ#norm/AssignMovingAvg/ReadVariableOpЂ*norm/AssignMovingAvg_1/AssignSubVariableOpЂ%norm/AssignMovingAvg_1/ReadVariableOpЂnorm/ReadVariableOpЂnorm/ReadVariableOp_1Ђoutput/BiasAdd/ReadVariableOpЂoutput/MatMul/ReadVariableOpЇ
input/Conv2D/ReadVariableOpReadVariableOp$input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
input/Conv2D/ReadVariableOpИ
input/Conv2DConv2Dinputs#input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџа*
paddingVALID*
strides
2
input/Conv2D
input/BiasAdd/ReadVariableOpReadVariableOp%input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
input/BiasAdd/ReadVariableOpЂ
input/BiasAddBiasAddinput/Conv2D:output:0$input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџа2
input/BiasAddt

input/ReluReluinput/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџа2

input/Reluh
norm/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
norm/LogicalAnd/xh
norm/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
norm/LogicalAnd/y
norm/LogicalAnd
LogicalAndnorm/LogicalAnd/x:output:0norm/LogicalAnd/y:output:0*
_output_shapes
: 2
norm/LogicalAnd
norm/ReadVariableOpReadVariableOpnorm_readvariableop_resource*
_output_shapes
:*
dtype02
norm/ReadVariableOp
norm/ReadVariableOp_1ReadVariableOpnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
norm/ReadVariableOp_1[

norm/ConstConst*
_output_shapes
: *
dtype0*
valueB 2

norm/Const_
norm/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
norm/Const_1З
norm/FusedBatchNormV3FusedBatchNormV3input/Relu:activations:0norm/ReadVariableOp:value:0norm/ReadVariableOp_1:value:0norm/Const:output:0norm/Const_1:output:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџа:::::*
epsilon%o:2
norm/FusedBatchNormV3a
norm/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
norm/Const_2Џ
norm/AssignMovingAvg/sub/xConst*0
_class&
$"loc:@norm/AssignMovingAvg/10873334*
_output_shapes
: *
dtype0*
valueB
 *  ?2
norm/AssignMovingAvg/sub/xЪ
norm/AssignMovingAvg/subSub#norm/AssignMovingAvg/sub/x:output:0norm/Const_2:output:0*
T0*0
_class&
$"loc:@norm/AssignMovingAvg/10873334*
_output_shapes
: 2
norm/AssignMovingAvg/subЄ
#norm/AssignMovingAvg/ReadVariableOpReadVariableOpnorm_assignmovingavg_10873334*
_output_shapes
:*
dtype02%
#norm/AssignMovingAvg/ReadVariableOpч
norm/AssignMovingAvg/sub_1Sub+norm/AssignMovingAvg/ReadVariableOp:value:0"norm/FusedBatchNormV3:batch_mean:0*
T0*0
_class&
$"loc:@norm/AssignMovingAvg/10873334*
_output_shapes
:2
norm/AssignMovingAvg/sub_1а
norm/AssignMovingAvg/mulMulnorm/AssignMovingAvg/sub_1:z:0norm/AssignMovingAvg/sub:z:0*
T0*0
_class&
$"loc:@norm/AssignMovingAvg/10873334*
_output_shapes
:2
norm/AssignMovingAvg/mulЃ
(norm/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpnorm_assignmovingavg_10873334norm/AssignMovingAvg/mul:z:0$^norm/AssignMovingAvg/ReadVariableOp*0
_class&
$"loc:@norm/AssignMovingAvg/10873334*
_output_shapes
 *
dtype02*
(norm/AssignMovingAvg/AssignSubVariableOpЕ
norm/AssignMovingAvg_1/sub/xConst*2
_class(
&$loc:@norm/AssignMovingAvg_1/10873341*
_output_shapes
: *
dtype0*
valueB
 *  ?2
norm/AssignMovingAvg_1/sub/xв
norm/AssignMovingAvg_1/subSub%norm/AssignMovingAvg_1/sub/x:output:0norm/Const_2:output:0*
T0*2
_class(
&$loc:@norm/AssignMovingAvg_1/10873341*
_output_shapes
: 2
norm/AssignMovingAvg_1/subЊ
%norm/AssignMovingAvg_1/ReadVariableOpReadVariableOpnorm_assignmovingavg_1_10873341*
_output_shapes
:*
dtype02'
%norm/AssignMovingAvg_1/ReadVariableOpѓ
norm/AssignMovingAvg_1/sub_1Sub-norm/AssignMovingAvg_1/ReadVariableOp:value:0&norm/FusedBatchNormV3:batch_variance:0*
T0*2
_class(
&$loc:@norm/AssignMovingAvg_1/10873341*
_output_shapes
:2
norm/AssignMovingAvg_1/sub_1к
norm/AssignMovingAvg_1/mulMul norm/AssignMovingAvg_1/sub_1:z:0norm/AssignMovingAvg_1/sub:z:0*
T0*2
_class(
&$loc:@norm/AssignMovingAvg_1/10873341*
_output_shapes
:2
norm/AssignMovingAvg_1/mulЏ
*norm/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpnorm_assignmovingavg_1_10873341norm/AssignMovingAvg_1/mul:z:0&^norm/AssignMovingAvg_1/ReadVariableOp*2
_class(
&$loc:@norm/AssignMovingAvg_1/10873341*
_output_shapes
 *
dtype02,
*norm/AssignMovingAvg_1/AssignSubVariableOpЊ
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
conv_2/Conv2D/ReadVariableOpЬ
conv_2/Conv2DConv2Dnorm/FusedBatchNormV3:y:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџfM	*
paddingVALID*
strides
2
conv_2/Conv2DЁ
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
conv_2/BiasAdd/ReadVariableOpЄ
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџfM	2
conv_2/BiasAddu
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџfM	2
conv_2/ReluЊ
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
conv_3/Conv2D/ReadVariableOpЬ
conv_3/Conv2DConv2Dconv_2/Relu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv_3/Conv2DЁ
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOpЄ
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv_3/BiasAddu
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv_3/Relui

flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ '  2

flat/Const
flat/ReshapeReshapeconv_3/Relu:activations:0flat/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџN2
flat/ReshapeЃ
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	N*
dtype02
dense1/MatMul/ReadVariableOp
dense1/MatMulMatMulflat/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense1/MatMulЁ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense1/BiasAdd/ReadVariableOp
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense1/BiasAddv
dense1/SigmoidSigmoiddense1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense1/SigmoidЂ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldense1/Sigmoid:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/MatMulЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/Softmaxљ
IdentityIdentityoutput/Softmax:softmax:0^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^input/BiasAdd/ReadVariableOp^input/Conv2D/ReadVariableOp)^norm/AssignMovingAvg/AssignSubVariableOp$^norm/AssignMovingAvg/ReadVariableOp+^norm/AssignMovingAvg_1/AssignSubVariableOp&^norm/AssignMovingAvg_1/ReadVariableOp^norm/ReadVariableOp^norm/ReadVariableOp_1^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2<
input/BiasAdd/ReadVariableOpinput/BiasAdd/ReadVariableOp2:
input/Conv2D/ReadVariableOpinput/Conv2D/ReadVariableOp2T
(norm/AssignMovingAvg/AssignSubVariableOp(norm/AssignMovingAvg/AssignSubVariableOp2J
#norm/AssignMovingAvg/ReadVariableOp#norm/AssignMovingAvg/ReadVariableOp2X
*norm/AssignMovingAvg_1/AssignSubVariableOp*norm/AssignMovingAvg_1/AssignSubVariableOp2N
%norm/AssignMovingAvg_1/ReadVariableOp%norm/AssignMovingAvg_1/ReadVariableOp2*
norm/ReadVariableOpnorm/ReadVariableOp2.
norm/ReadVariableOp_1norm/ReadVariableOp_12>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ж&

J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873162
input_input(
$input_statefulpartitionedcall_args_1(
$input_statefulpartitionedcall_args_2'
#norm_statefulpartitionedcall_args_1'
#norm_statefulpartitionedcall_args_2'
#norm_statefulpartitionedcall_args_3'
#norm_statefulpartitionedcall_args_4)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2)
%dense1_statefulpartitionedcall_args_1)
%dense1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identityЂconv_2/StatefulPartitionedCallЂconv_3/StatefulPartitionedCallЂdense1/StatefulPartitionedCallЂinput/StatefulPartitionedCallЂnorm/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЌ
input/StatefulPartitionedCallStatefulPartitionedCallinput_input$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:џџџџџџџџџа**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_108728272
input/StatefulPartitionedCall
norm/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0#norm_statefulpartitionedcall_args_1#norm_statefulpartitionedcall_args_2#norm_statefulpartitionedcall_args_3#norm_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:џџџџџџџџџа**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_norm_layer_call_and_return_conditional_losses_108730492
norm/StatefulPartitionedCallЩ
conv_2/StatefulPartitionedCallStatefulPartitionedCall%norm/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџfM	**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_108729802 
conv_2/StatefulPartitionedCallЫ
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџ **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_108730012 
conv_3/StatefulPartitionedCallж
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:џџџџџџџџџN**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_108731072
flat/PartitionedCallЙ
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_108731262 
dense1/StatefulPartitionedCallУ
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_108731492 
output/StatefulPartitionedCallО
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^norm/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2<
norm/StatefulPartitionedCallnorm/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
$

B__inference_norm_layer_call_and_return_conditional_losses_10873049

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_10873034
assignmovingavg_1_10873041
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџа:::::*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2	
Const_2 
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/10873034*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xБ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10873034*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10873034*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpЮ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/10873034*
_output_shapes
:2
AssignMovingAvg/sub_1З
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/10873034*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10873034AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/10873034*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpІ
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/10873041*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЙ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10873041*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10873041*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10873041*
_output_shapes
:2
AssignMovingAvg_1/sub_1С
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10873041*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10873041AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/10873041*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЈ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:џџџџџџџџџа2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџа::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
ў
х
/__inference_gen_60_ind_3_layer_call_fn_10873476

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
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_108732592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Щ	
н
D__inference_dense1_layer_call_and_return_conditional_losses_10873658

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	N*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџN::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ї&
ќ
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873259

inputs(
$input_statefulpartitionedcall_args_1(
$input_statefulpartitionedcall_args_2'
#norm_statefulpartitionedcall_args_1'
#norm_statefulpartitionedcall_args_2'
#norm_statefulpartitionedcall_args_3'
#norm_statefulpartitionedcall_args_4)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2)
%dense1_statefulpartitionedcall_args_1)
%dense1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identityЂconv_2/StatefulPartitionedCallЂconv_3/StatefulPartitionedCallЂdense1/StatefulPartitionedCallЂinput/StatefulPartitionedCallЂnorm/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЇ
input/StatefulPartitionedCallStatefulPartitionedCallinputs$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:џџџџџџџџџа**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_108728272
input/StatefulPartitionedCall
norm/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0#norm_statefulpartitionedcall_args_1#norm_statefulpartitionedcall_args_2#norm_statefulpartitionedcall_args_3#norm_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:џџџџџџџџџа**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_norm_layer_call_and_return_conditional_losses_108730712
norm/StatefulPartitionedCallЩ
conv_2/StatefulPartitionedCallStatefulPartitionedCall%norm/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџfM	**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_108729802 
conv_2/StatefulPartitionedCallЫ
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџ **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_108730012 
conv_3/StatefulPartitionedCallж
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:џџџџџџџџџN**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_108731072
flat/PartitionedCallЙ
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_108731262 
dense1/StatefulPartitionedCallУ
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_108731492 
output/StatefulPartitionedCallО
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^norm/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2<
norm/StatefulPartitionedCallnorm/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
B
џ
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873438

inputs(
$input_conv2d_readvariableop_resource)
%input_biasadd_readvariableop_resource 
norm_readvariableop_resource"
norm_readvariableop_1_resource1
-norm_fusedbatchnormv3_readvariableop_resource3
/norm_fusedbatchnormv3_readvariableop_1_resource)
%conv_2_conv2d_readvariableop_resource*
&conv_2_biasadd_readvariableop_resource)
%conv_3_conv2d_readvariableop_resource*
&conv_3_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityЂconv_2/BiasAdd/ReadVariableOpЂconv_2/Conv2D/ReadVariableOpЂconv_3/BiasAdd/ReadVariableOpЂconv_3/Conv2D/ReadVariableOpЂdense1/BiasAdd/ReadVariableOpЂdense1/MatMul/ReadVariableOpЂinput/BiasAdd/ReadVariableOpЂinput/Conv2D/ReadVariableOpЂ$norm/FusedBatchNormV3/ReadVariableOpЂ&norm/FusedBatchNormV3/ReadVariableOp_1Ђnorm/ReadVariableOpЂnorm/ReadVariableOp_1Ђoutput/BiasAdd/ReadVariableOpЂoutput/MatMul/ReadVariableOpЇ
input/Conv2D/ReadVariableOpReadVariableOp$input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
input/Conv2D/ReadVariableOpИ
input/Conv2DConv2Dinputs#input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџа*
paddingVALID*
strides
2
input/Conv2D
input/BiasAdd/ReadVariableOpReadVariableOp%input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
input/BiasAdd/ReadVariableOpЂ
input/BiasAddBiasAddinput/Conv2D:output:0$input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџа2
input/BiasAddt

input/ReluReluinput/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџа2

input/Reluh
norm/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
norm/LogicalAnd/xh
norm/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
norm/LogicalAnd/y
norm/LogicalAnd
LogicalAndnorm/LogicalAnd/x:output:0norm/LogicalAnd/y:output:0*
_output_shapes
: 2
norm/LogicalAnd
norm/ReadVariableOpReadVariableOpnorm_readvariableop_resource*
_output_shapes
:*
dtype02
norm/ReadVariableOp
norm/ReadVariableOp_1ReadVariableOpnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
norm/ReadVariableOp_1Ж
$norm/FusedBatchNormV3/ReadVariableOpReadVariableOp-norm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02&
$norm/FusedBatchNormV3/ReadVariableOpМ
&norm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/norm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&norm/FusedBatchNormV3/ReadVariableOp_1ќ
norm/FusedBatchNormV3FusedBatchNormV3input/Relu:activations:0norm/ReadVariableOp:value:0norm/ReadVariableOp_1:value:0,norm/FusedBatchNormV3/ReadVariableOp:value:0.norm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџа:::::*
epsilon%o:*
is_training( 2
norm/FusedBatchNormV3]

norm/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2

norm/ConstЊ
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
conv_2/Conv2D/ReadVariableOpЬ
conv_2/Conv2DConv2Dnorm/FusedBatchNormV3:y:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџfM	*
paddingVALID*
strides
2
conv_2/Conv2DЁ
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
conv_2/BiasAdd/ReadVariableOpЄ
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџfM	2
conv_2/BiasAddu
conv_2/ReluReluconv_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџfM	2
conv_2/ReluЊ
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
conv_3/Conv2D/ReadVariableOpЬ
conv_3/Conv2DConv2Dconv_2/Relu:activations:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
conv_3/Conv2DЁ
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOpЄ
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv_3/BiasAddu
conv_3/ReluReluconv_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv_3/Relui

flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ '  2

flat/Const
flat/ReshapeReshapeconv_3/Relu:activations:0flat/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџN2
flat/ReshapeЃ
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	N*
dtype02
dense1/MatMul/ReadVariableOp
dense1/MatMulMatMulflat/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense1/MatMulЁ
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense1/BiasAdd/ReadVariableOp
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense1/BiasAddv
dense1/SigmoidSigmoiddense1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense1/SigmoidЂ
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldense1/Sigmoid:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/MatMulЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
output/SoftmaxЃ
IdentityIdentityoutput/Softmax:softmax:0^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^input/BiasAdd/ReadVariableOp^input/Conv2D/ReadVariableOp%^norm/FusedBatchNormV3/ReadVariableOp'^norm/FusedBatchNormV3/ReadVariableOp_1^norm/ReadVariableOp^norm/ReadVariableOp_1^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2<
input/BiasAdd/ReadVariableOpinput/BiasAdd/ReadVariableOp2:
input/Conv2D/ReadVariableOpinput/Conv2D/ReadVariableOp2L
$norm/FusedBatchNormV3/ReadVariableOp$norm/FusedBatchNormV3/ReadVariableOp2P
&norm/FusedBatchNormV3/ReadVariableOp_1&norm/FusedBatchNormV3/ReadVariableOp_12*
norm/ReadVariableOpnorm/ReadVariableOp2.
norm/ReadVariableOp_1norm/ReadVariableOp_12>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Р
Њ
)__inference_conv_2_layer_call_fn_10872988

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_108729802
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ђ
х
B__inference_norm_layer_call_and_return_conditional_losses_10873544

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Р
Њ
)__inference_conv_3_layer_call_fn_10873009

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_108730012
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
$

B__inference_norm_layer_call_and_return_conditional_losses_10873596

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_10873581
assignmovingavg_1_10873588
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџа:::::*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2	
Const_2 
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/10873581*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xБ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10873581*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10873581*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpЮ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/10873581*
_output_shapes
:2
AssignMovingAvg/sub_1З
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/10873581*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10873581AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/10873581*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpІ
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/10873588*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЙ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10873588*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10873588*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10873588*
_output_shapes
:2
AssignMovingAvg_1/sub_1С
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10873588*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10873588AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/10873588*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpЈ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:џџџџџџџџџа2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџа::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
н
с
&__inference_signature_wrapper_10873308
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
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinput_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*,
f'R%
#__inference__wrapped_model_108728142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_input

^
B__inference_flat_layer_call_and_return_conditional_losses_10873107

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ '  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџN2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
ђ
х
B__inference_norm_layer_call_and_return_conditional_losses_10872960

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
Constь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs

ъ
/__inference_gen_60_ind_3_layer_call_fn_10873232
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
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identityЂStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinput_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_108732152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
Ы$

B__inference_norm_layer_call_and_return_conditional_losses_10873522

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_10873507
assignmovingavg_1_10873514
identityЂ#AssignMovingAvg/AssignSubVariableOpЂAssignMovingAvg/ReadVariableOpЂ%AssignMovingAvg_1/AssignSubVariableOpЂ AssignMovingAvg_1/ReadVariableOpЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2	
Const_2 
AssignMovingAvg/sub/xConst*+
_class!
loc:@AssignMovingAvg/10873507*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg/sub/xБ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*+
_class!
loc:@AssignMovingAvg/10873507*
_output_shapes
: 2
AssignMovingAvg/sub
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_10873507*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpЮ
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*+
_class!
loc:@AssignMovingAvg/10873507*
_output_shapes
:2
AssignMovingAvg/sub_1З
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*+
_class!
loc:@AssignMovingAvg/10873507*
_output_shapes
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_10873507AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*+
_class!
loc:@AssignMovingAvg/10873507*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpІ
AssignMovingAvg_1/sub/xConst*-
_class#
!loc:@AssignMovingAvg_1/10873514*
_output_shapes
: *
dtype0*
valueB
 *  ?2
AssignMovingAvg_1/sub/xЙ
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10873514*
_output_shapes
: 2
AssignMovingAvg_1/sub
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_10873514*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpк
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10873514*
_output_shapes
:2
AssignMovingAvg_1/sub_1С
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0*-
_class#
!loc:@AssignMovingAvg_1/10873514*
_output_shapes
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_10873514AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*-
_class#
!loc:@AssignMovingAvg_1/10873514*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpИ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Ю
№
'__inference_norm_layer_call_fn_10873553

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_norm_layer_call_and_return_conditional_losses_108729292
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
щ
н
D__inference_conv_3_layer_call_and_return_conditional_losses_10873001

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
Т
х
B__inference_norm_layer_call_and_return_conditional_losses_10873618

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ь
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџа:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
Constм
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:џџџџџџџџџа2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџа::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
Э	
н
D__inference_output_layer_call_and_return_conditional_losses_10873676

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Ж&

J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873187
input_input(
$input_statefulpartitionedcall_args_1(
$input_statefulpartitionedcall_args_2'
#norm_statefulpartitionedcall_args_1'
#norm_statefulpartitionedcall_args_2'
#norm_statefulpartitionedcall_args_3'
#norm_statefulpartitionedcall_args_4)
%conv_2_statefulpartitionedcall_args_1)
%conv_2_statefulpartitionedcall_args_2)
%conv_3_statefulpartitionedcall_args_1)
%conv_3_statefulpartitionedcall_args_2)
%dense1_statefulpartitionedcall_args_1)
%dense1_statefulpartitionedcall_args_2)
%output_statefulpartitionedcall_args_1)
%output_statefulpartitionedcall_args_2
identityЂconv_2/StatefulPartitionedCallЂconv_3/StatefulPartitionedCallЂdense1/StatefulPartitionedCallЂinput/StatefulPartitionedCallЂnorm/StatefulPartitionedCallЂoutput/StatefulPartitionedCallЌ
input/StatefulPartitionedCallStatefulPartitionedCallinput_input$input_statefulpartitionedcall_args_1$input_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:џџџџџџџџџа**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_input_layer_call_and_return_conditional_losses_108728272
input/StatefulPartitionedCall
norm/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0#norm_statefulpartitionedcall_args_1#norm_statefulpartitionedcall_args_2#norm_statefulpartitionedcall_args_3#norm_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:џџџџџџџџџа**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_norm_layer_call_and_return_conditional_losses_108730712
norm/StatefulPartitionedCallЩ
conv_2/StatefulPartitionedCallStatefulPartitionedCall%norm/StatefulPartitionedCall:output:0%conv_2_statefulpartitionedcall_args_1%conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџfM	**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv_2_layer_call_and_return_conditional_losses_108729802 
conv_2/StatefulPartitionedCallЫ
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0%conv_3_statefulpartitionedcall_args_1%conv_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:џџџџџџџџџ **
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_conv_3_layer_call_and_return_conditional_losses_108730012 
conv_3/StatefulPartitionedCallж
flat/PartitionedCallPartitionedCall'conv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:џџџџџџџџџN**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_flat_layer_call_and_return_conditional_losses_108731072
flat/PartitionedCallЙ
dense1/StatefulPartitionedCallStatefulPartitionedCallflat/PartitionedCall:output:0%dense1_statefulpartitionedcall_args_1%dense1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_108731262 
dense1/StatefulPartitionedCallУ
output/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0%output_statefulpartitionedcall_args_1%output_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_108731492 
output/StatefulPartitionedCallО
IdentityIdentity'output/StatefulPartitionedCall:output:0^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^input/StatefulPartitionedCall^norm/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2<
norm/StatefulPartitionedCallnorm/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:+ '
%
_user_specified_nameinput_input
ЅU
Щ
#__inference__wrapped_model_10872814
input_input5
1gen_60_ind_3_input_conv2d_readvariableop_resource6
2gen_60_ind_3_input_biasadd_readvariableop_resource-
)gen_60_ind_3_norm_readvariableop_resource/
+gen_60_ind_3_norm_readvariableop_1_resource>
:gen_60_ind_3_norm_fusedbatchnormv3_readvariableop_resource@
<gen_60_ind_3_norm_fusedbatchnormv3_readvariableop_1_resource6
2gen_60_ind_3_conv_2_conv2d_readvariableop_resource7
3gen_60_ind_3_conv_2_biasadd_readvariableop_resource6
2gen_60_ind_3_conv_3_conv2d_readvariableop_resource7
3gen_60_ind_3_conv_3_biasadd_readvariableop_resource6
2gen_60_ind_3_dense1_matmul_readvariableop_resource7
3gen_60_ind_3_dense1_biasadd_readvariableop_resource6
2gen_60_ind_3_output_matmul_readvariableop_resource7
3gen_60_ind_3_output_biasadd_readvariableop_resource
identityЂ*gen_60_ind_3/conv_2/BiasAdd/ReadVariableOpЂ)gen_60_ind_3/conv_2/Conv2D/ReadVariableOpЂ*gen_60_ind_3/conv_3/BiasAdd/ReadVariableOpЂ)gen_60_ind_3/conv_3/Conv2D/ReadVariableOpЂ*gen_60_ind_3/dense1/BiasAdd/ReadVariableOpЂ)gen_60_ind_3/dense1/MatMul/ReadVariableOpЂ)gen_60_ind_3/input/BiasAdd/ReadVariableOpЂ(gen_60_ind_3/input/Conv2D/ReadVariableOpЂ1gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOpЂ3gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOp_1Ђ gen_60_ind_3/norm/ReadVariableOpЂ"gen_60_ind_3/norm/ReadVariableOp_1Ђ*gen_60_ind_3/output/BiasAdd/ReadVariableOpЂ)gen_60_ind_3/output/MatMul/ReadVariableOpЮ
(gen_60_ind_3/input/Conv2D/ReadVariableOpReadVariableOp1gen_60_ind_3_input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(gen_60_ind_3/input/Conv2D/ReadVariableOpф
gen_60_ind_3/input/Conv2DConv2Dinput_input0gen_60_ind_3/input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџа*
paddingVALID*
strides
2
gen_60_ind_3/input/Conv2DХ
)gen_60_ind_3/input/BiasAdd/ReadVariableOpReadVariableOp2gen_60_ind_3_input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)gen_60_ind_3/input/BiasAdd/ReadVariableOpж
gen_60_ind_3/input/BiasAddBiasAdd"gen_60_ind_3/input/Conv2D:output:01gen_60_ind_3/input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџа2
gen_60_ind_3/input/BiasAdd
gen_60_ind_3/input/ReluRelu#gen_60_ind_3/input/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџа2
gen_60_ind_3/input/Relu
gen_60_ind_3/norm/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2 
gen_60_ind_3/norm/LogicalAnd/x
gen_60_ind_3/norm/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2 
gen_60_ind_3/norm/LogicalAnd/yД
gen_60_ind_3/norm/LogicalAnd
LogicalAnd'gen_60_ind_3/norm/LogicalAnd/x:output:0'gen_60_ind_3/norm/LogicalAnd/y:output:0*
_output_shapes
: 2
gen_60_ind_3/norm/LogicalAndЊ
 gen_60_ind_3/norm/ReadVariableOpReadVariableOp)gen_60_ind_3_norm_readvariableop_resource*
_output_shapes
:*
dtype02"
 gen_60_ind_3/norm/ReadVariableOpА
"gen_60_ind_3/norm/ReadVariableOp_1ReadVariableOp+gen_60_ind_3_norm_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"gen_60_ind_3/norm/ReadVariableOp_1н
1gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOpReadVariableOp:gen_60_ind_3_norm_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOpу
3gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<gen_60_ind_3_norm_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOp_1з
"gen_60_ind_3/norm/FusedBatchNormV3FusedBatchNormV3%gen_60_ind_3/input/Relu:activations:0(gen_60_ind_3/norm/ReadVariableOp:value:0*gen_60_ind_3/norm/ReadVariableOp_1:value:09gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOp:value:0;gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:џџџџџџџџџа:::::*
epsilon%o:*
is_training( 2$
"gen_60_ind_3/norm/FusedBatchNormV3w
gen_60_ind_3/norm/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Єp}?2
gen_60_ind_3/norm/Constб
)gen_60_ind_3/conv_2/Conv2D/ReadVariableOpReadVariableOp2gen_60_ind_3_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02+
)gen_60_ind_3/conv_2/Conv2D/ReadVariableOp
gen_60_ind_3/conv_2/Conv2DConv2D&gen_60_ind_3/norm/FusedBatchNormV3:y:01gen_60_ind_3/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџfM	*
paddingVALID*
strides
2
gen_60_ind_3/conv_2/Conv2DШ
*gen_60_ind_3/conv_2/BiasAdd/ReadVariableOpReadVariableOp3gen_60_ind_3_conv_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02,
*gen_60_ind_3/conv_2/BiasAdd/ReadVariableOpи
gen_60_ind_3/conv_2/BiasAddBiasAdd#gen_60_ind_3/conv_2/Conv2D:output:02gen_60_ind_3/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџfM	2
gen_60_ind_3/conv_2/BiasAdd
gen_60_ind_3/conv_2/ReluRelu$gen_60_ind_3/conv_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџfM	2
gen_60_ind_3/conv_2/Reluб
)gen_60_ind_3/conv_3/Conv2D/ReadVariableOpReadVariableOp2gen_60_ind_3_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02+
)gen_60_ind_3/conv_3/Conv2D/ReadVariableOp
gen_60_ind_3/conv_3/Conv2DConv2D&gen_60_ind_3/conv_2/Relu:activations:01gen_60_ind_3/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
gen_60_ind_3/conv_3/Conv2DШ
*gen_60_ind_3/conv_3/BiasAdd/ReadVariableOpReadVariableOp3gen_60_ind_3_conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*gen_60_ind_3/conv_3/BiasAdd/ReadVariableOpи
gen_60_ind_3/conv_3/BiasAddBiasAdd#gen_60_ind_3/conv_3/Conv2D:output:02gen_60_ind_3/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
gen_60_ind_3/conv_3/BiasAdd
gen_60_ind_3/conv_3/ReluRelu$gen_60_ind_3/conv_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
gen_60_ind_3/conv_3/Relu
gen_60_ind_3/flat/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ '  2
gen_60_ind_3/flat/ConstО
gen_60_ind_3/flat/ReshapeReshape&gen_60_ind_3/conv_3/Relu:activations:0 gen_60_ind_3/flat/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџN2
gen_60_ind_3/flat/ReshapeЪ
)gen_60_ind_3/dense1/MatMul/ReadVariableOpReadVariableOp2gen_60_ind_3_dense1_matmul_readvariableop_resource*
_output_shapes
:	N*
dtype02+
)gen_60_ind_3/dense1/MatMul/ReadVariableOpЫ
gen_60_ind_3/dense1/MatMulMatMul"gen_60_ind_3/flat/Reshape:output:01gen_60_ind_3/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gen_60_ind_3/dense1/MatMulШ
*gen_60_ind_3/dense1/BiasAdd/ReadVariableOpReadVariableOp3gen_60_ind_3_dense1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*gen_60_ind_3/dense1/BiasAdd/ReadVariableOpб
gen_60_ind_3/dense1/BiasAddBiasAdd$gen_60_ind_3/dense1/MatMul:product:02gen_60_ind_3/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gen_60_ind_3/dense1/BiasAdd
gen_60_ind_3/dense1/SigmoidSigmoid$gen_60_ind_3/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gen_60_ind_3/dense1/SigmoidЩ
)gen_60_ind_3/output/MatMul/ReadVariableOpReadVariableOp2gen_60_ind_3_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)gen_60_ind_3/output/MatMul/ReadVariableOpШ
gen_60_ind_3/output/MatMulMatMulgen_60_ind_3/dense1/Sigmoid:y:01gen_60_ind_3/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gen_60_ind_3/output/MatMulШ
*gen_60_ind_3/output/BiasAdd/ReadVariableOpReadVariableOp3gen_60_ind_3_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*gen_60_ind_3/output/BiasAdd/ReadVariableOpб
gen_60_ind_3/output/BiasAddBiasAdd$gen_60_ind_3/output/MatMul:product:02gen_60_ind_3/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gen_60_ind_3/output/BiasAdd
gen_60_ind_3/output/SoftmaxSoftmax$gen_60_ind_3/output/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
gen_60_ind_3/output/Softmaxц
IdentityIdentity%gen_60_ind_3/output/Softmax:softmax:0+^gen_60_ind_3/conv_2/BiasAdd/ReadVariableOp*^gen_60_ind_3/conv_2/Conv2D/ReadVariableOp+^gen_60_ind_3/conv_3/BiasAdd/ReadVariableOp*^gen_60_ind_3/conv_3/Conv2D/ReadVariableOp+^gen_60_ind_3/dense1/BiasAdd/ReadVariableOp*^gen_60_ind_3/dense1/MatMul/ReadVariableOp*^gen_60_ind_3/input/BiasAdd/ReadVariableOp)^gen_60_ind_3/input/Conv2D/ReadVariableOp2^gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOp4^gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOp_1!^gen_60_ind_3/norm/ReadVariableOp#^gen_60_ind_3/norm/ReadVariableOp_1+^gen_60_ind_3/output/BiasAdd/ReadVariableOp*^gen_60_ind_3/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::2X
*gen_60_ind_3/conv_2/BiasAdd/ReadVariableOp*gen_60_ind_3/conv_2/BiasAdd/ReadVariableOp2V
)gen_60_ind_3/conv_2/Conv2D/ReadVariableOp)gen_60_ind_3/conv_2/Conv2D/ReadVariableOp2X
*gen_60_ind_3/conv_3/BiasAdd/ReadVariableOp*gen_60_ind_3/conv_3/BiasAdd/ReadVariableOp2V
)gen_60_ind_3/conv_3/Conv2D/ReadVariableOp)gen_60_ind_3/conv_3/Conv2D/ReadVariableOp2X
*gen_60_ind_3/dense1/BiasAdd/ReadVariableOp*gen_60_ind_3/dense1/BiasAdd/ReadVariableOp2V
)gen_60_ind_3/dense1/MatMul/ReadVariableOp)gen_60_ind_3/dense1/MatMul/ReadVariableOp2V
)gen_60_ind_3/input/BiasAdd/ReadVariableOp)gen_60_ind_3/input/BiasAdd/ReadVariableOp2T
(gen_60_ind_3/input/Conv2D/ReadVariableOp(gen_60_ind_3/input/Conv2D/ReadVariableOp2f
1gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOp1gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOp2j
3gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOp_13gen_60_ind_3/norm/FusedBatchNormV3/ReadVariableOp_12D
 gen_60_ind_3/norm/ReadVariableOp gen_60_ind_3/norm/ReadVariableOp2H
"gen_60_ind_3/norm/ReadVariableOp_1"gen_60_ind_3/norm/ReadVariableOp_12X
*gen_60_ind_3/output/BiasAdd/ReadVariableOp*gen_60_ind_3/output/BiasAdd/ReadVariableOp2V
)gen_60_ind_3/output/MatMul/ReadVariableOp)gen_60_ind_3/output/MatMul/ReadVariableOp:+ '
%
_user_specified_nameinput_input
ў
х
/__inference_gen_60_ind_3_layer_call_fn_10873457

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
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:џџџџџџџџџ**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_108732152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*h
_input_shapesW
U:џџџџџџџџџв ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs

^
B__inference_flat_layer_call_and_return_conditional_losses_10873642

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ '  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџN2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџN2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ :& "
 
_user_specified_nameinputs
Ю(
Љ
!__inference__traced_save_10873749
file_prefix.
*savev2_input_87_kernel_read_readvariableop,
(savev2_input_87_bias_read_readvariableop,
(savev2_norm_87_gamma_read_readvariableop+
'savev2_norm_87_beta_read_readvariableop2
.savev2_norm_87_moving_mean_read_readvariableop6
2savev2_norm_87_moving_variance_read_readvariableop/
+savev2_conv_2_87_kernel_read_readvariableop-
)savev2_conv_2_87_bias_read_readvariableop/
+savev2_conv_3_87_kernel_read_readvariableop-
)savev2_conv_3_87_bias_read_readvariableop/
+savev2_dense1_87_kernel_read_readvariableop-
)savev2_dense1_87_bias_read_readvariableop/
+savev2_output_87_kernel_read_readvariableop-
)savev2_output_87_bias_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1Ѕ
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_69f2ce2ae51e412dad2a05acfa3549e1/part2
StringJoin/inputs_1

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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*І
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesЄ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesІ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_input_87_kernel_read_readvariableop(savev2_input_87_bias_read_readvariableop(savev2_norm_87_gamma_read_readvariableop'savev2_norm_87_beta_read_readvariableop.savev2_norm_87_moving_mean_read_readvariableop2savev2_norm_87_moving_variance_read_readvariableop+savev2_conv_2_87_kernel_read_readvariableop)savev2_conv_2_87_bias_read_readvariableop+savev2_conv_3_87_kernel_read_readvariableop)savev2_conv_3_87_bias_read_readvariableop+savev2_dense1_87_kernel_read_readvariableop)savev2_dense1_87_bias_read_readvariableop+savev2_output_87_kernel_read_readvariableop)savev2_output_87_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardЌ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ђ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesЯ
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
&MergeV2Checkpoints/checkpoint_prefixesЌ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :::::::	:	:	::	N:::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix

№
'__inference_norm_layer_call_fn_10873636

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*1
_output_shapes
:џџџџџџџџџа**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_norm_layer_call_and_return_conditional_losses_108730712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџа2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:џџџџџџџџџа::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"ЏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Л
serving_defaultЇ
M
input_input>
serving_default_input_input:0џџџџџџџџџв :
output0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:щ
к9
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		variables

regularization_losses
trainable_variables
	keras_api

signatures
Y_default_save_signature
*Z&call_and_return_all_conditional_losses
[__call__"6
_tf_keras_sequentialћ5{"class_name": "Sequential", "name": "gen_60_ind_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gen_60_ind_3", "layers": [{"class_name": "Conv2D", "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 1], "dtype": "float32", "filters": 7, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "norm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 13, "kernel_size": [7, 7], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 25, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "gen_60_ind_3", "layers": [{"class_name": "Conv2D", "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 1], "dtype": "float32", "filters": 7, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "norm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 13, "kernel_size": [7, 7], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 25, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
Й"Ж
_tf_keras_input_layer{"class_name": "InputLayer", "name": "input_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 210, 160, 1], "config": {"batch_input_shape": [null, 210, 160, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_input"}}


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"љ
_tf_keras_layerп{"class_name": "Conv2D", "name": "input", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 210, 160, 1], "config": {"name": "input", "trainable": true, "batch_input_shape": [null, 210, 160, 1], "dtype": "float32", "filters": 7, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}

axis
	gamma
beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
*^&call_and_return_all_conditional_losses
___call__"М
_tf_keras_layerЂ{"class_name": "BatchNormalization", "name": "norm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "norm", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 7}}}}
ч

kernel
bias
	variables
 regularization_losses
!trainable_variables
"	keras_api
*`&call_and_return_all_conditional_losses
a__call__"Т
_tf_keras_layerЈ{"class_name": "Conv2D", "name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": [5, 5], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 7}}}}
ш

#kernel
$bias
%	variables
&regularization_losses
'trainable_variables
(	keras_api
*b&call_and_return_all_conditional_losses
c__call__"У
_tf_keras_layerЉ{"class_name": "Conv2D", "name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 13, "kernel_size": [7, 7], "strides": [3, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 9}}}}
І
)	variables
*regularization_losses
+trainable_variables
,	keras_api
*d&call_and_return_all_conditional_losses
e__call__"
_tf_keras_layer§{"class_name": "Flatten", "name": "flat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flat", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
є

-kernel
.bias
/	variables
0regularization_losses
1trainable_variables
2	keras_api
*f&call_and_return_all_conditional_losses
g__call__"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 25, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9984}}}}
ё

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
*h&call_and_return_all_conditional_losses
i__call__"Ь
_tf_keras_layerВ{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 25}}}}

0
1
2
3
4
5
6
7
#8
$9
-10
.11
312
413"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
#6
$7
-8
.9
310
411"
trackable_list_wrapper
З
9metrics
		variables

regularization_losses
:non_trainable_variables
;layer_regularization_losses

<layers
trainable_variables
[__call__
Y_default_save_signature
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
,
jserving_default"
signature_map
):'2input_87/kernel
:2input_87/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

=metrics
	variables
regularization_losses
>non_trainable_variables
?layer_regularization_losses

@layers
trainable_variables
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2norm_87/gamma
:2norm_87/beta
#:! (2norm_87/moving_mean
':% (2norm_87/moving_variance
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

Ametrics
	variables
regularization_losses
Bnon_trainable_variables
Clayer_regularization_losses

Dlayers
trainable_variables
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
*:(	2conv_2_87/kernel
:	2conv_2_87/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

Emetrics
	variables
 regularization_losses
Fnon_trainable_variables
Glayer_regularization_losses

Hlayers
!trainable_variables
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
*:(	2conv_3_87/kernel
:2conv_3_87/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper

Imetrics
%	variables
&regularization_losses
Jnon_trainable_variables
Klayer_regularization_losses

Llayers
'trainable_variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

Mmetrics
)	variables
*regularization_losses
Nnon_trainable_variables
Olayer_regularization_losses

Players
+trainable_variables
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
#:!	N2dense1_87/kernel
:2dense1_87/bias
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper

Qmetrics
/	variables
0regularization_losses
Rnon_trainable_variables
Slayer_regularization_losses

Tlayers
1trainable_variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
": 2output_87/kernel
:2output_87/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper

Umetrics
5	variables
6regularization_losses
Vnon_trainable_variables
Wlayer_regularization_losses

Xlayers
7trainable_variables
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
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
.
0
1"
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
я2ь
#__inference__wrapped_model_10872814Ф
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *4Ђ1
/,
input_inputџџџџџџџџџв 
і2ѓ
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873162
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873379
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873438
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873187Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
/__inference_gen_60_ind_3_layer_call_fn_10873232
/__inference_gen_60_ind_3_layer_call_fn_10873476
/__inference_gen_60_ind_3_layer_call_fn_10873276
/__inference_gen_60_ind_3_layer_call_fn_10873457Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ђ2
C__inference_input_layer_call_and_return_conditional_losses_10872827з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
(__inference_input_layer_call_fn_10872835з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ъ2Ч
B__inference_norm_layer_call_and_return_conditional_losses_10873522
B__inference_norm_layer_call_and_return_conditional_losses_10873544
B__inference_norm_layer_call_and_return_conditional_losses_10873596
B__inference_norm_layer_call_and_return_conditional_losses_10873618Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
'__inference_norm_layer_call_fn_10873553
'__inference_norm_layer_call_fn_10873627
'__inference_norm_layer_call_fn_10873562
'__inference_norm_layer_call_fn_10873636Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ѓ2 
D__inference_conv_2_layer_call_and_return_conditional_losses_10872980з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
)__inference_conv_2_layer_call_fn_10872988з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ѓ2 
D__inference_conv_3_layer_call_and_return_conditional_losses_10873001з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
2
)__inference_conv_3_layer_call_fn_10873009з
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
ь2щ
B__inference_flat_layer_call_and_return_conditional_losses_10873642Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_flat_layer_call_fn_10873647Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_dense1_layer_call_and_return_conditional_losses_10873658Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_dense1_layer_call_fn_10873665Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_output_layer_call_and_return_conditional_losses_10873676Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_output_layer_call_fn_10873683Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
9B7
&__inference_signature_wrapper_10873308input_inputЉ
#__inference__wrapped_model_10872814#$-.34>Ђ;
4Ђ1
/,
input_inputџџџџџџџџџв 
Њ "/Њ,
*
output 
outputџџџџџџџџџй
D__inference_conv_2_layer_call_and_return_conditional_losses_10872980IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
 Б
)__inference_conv_2_layer_call_fn_10872988IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	й
D__inference_conv_3_layer_call_and_return_conditional_losses_10873001#$IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Б
)__inference_conv_3_layer_call_fn_10873009#$IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ	
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
D__inference_dense1_layer_call_and_return_conditional_losses_10873658]-.0Ђ-
&Ђ#
!
inputsџџџџџџџџџN
Њ "%Ђ"

0џџџџџџџџџ
 }
)__inference_dense1_layer_call_fn_10873665P-.0Ђ-
&Ђ#
!
inputsџџџџџџџџџN
Њ "џџџџџџџџџЇ
B__inference_flat_layer_call_and_return_conditional_losses_10873642a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "&Ђ#

0џџџџџџџџџN
 
'__inference_flat_layer_call_fn_10873647T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "џџџџџџџџџNЭ
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873162#$-.34FЂC
<Ђ9
/,
input_inputџџџџџџџџџв 
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Э
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873187#$-.34FЂC
<Ђ9
/,
input_inputџџџџџџџџџв 
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ш
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873379z#$-.34AЂ>
7Ђ4
*'
inputsџџџџџџџџџв 
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ш
J__inference_gen_60_ind_3_layer_call_and_return_conditional_losses_10873438z#$-.34AЂ>
7Ђ4
*'
inputsџџџџџџџџџв 
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ѕ
/__inference_gen_60_ind_3_layer_call_fn_10873232r#$-.34FЂC
<Ђ9
/,
input_inputџџџџџџџџџв 
p

 
Њ "џџџџџџџџџЅ
/__inference_gen_60_ind_3_layer_call_fn_10873276r#$-.34FЂC
<Ђ9
/,
input_inputџџџџџџџџџв 
p 

 
Њ "џџџџџџџџџ 
/__inference_gen_60_ind_3_layer_call_fn_10873457m#$-.34AЂ>
7Ђ4
*'
inputsџџџџџџџџџв 
p

 
Њ "џџџџџџџџџ 
/__inference_gen_60_ind_3_layer_call_fn_10873476m#$-.34AЂ>
7Ђ4
*'
inputsџџџџџџџџџв 
p 

 
Њ "џџџџџџџџџи
C__inference_input_layer_call_and_return_conditional_losses_10872827IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 А
(__inference_input_layer_call_fn_10872835IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџн
B__inference_norm_layer_call_and_return_conditional_losses_10873522MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 н
B__inference_norm_layer_call_and_return_conditional_losses_10873544MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 М
B__inference_norm_layer_call_and_return_conditional_losses_10873596v=Ђ:
3Ђ0
*'
inputsџџџџџџџџџа
p
Њ "/Ђ,
%"
0џџџџџџџџџа
 М
B__inference_norm_layer_call_and_return_conditional_losses_10873618v=Ђ:
3Ђ0
*'
inputsџџџџџџџџџа
p 
Њ "/Ђ,
%"
0џџџџџџџџџа
 Е
'__inference_norm_layer_call_fn_10873553MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЕ
'__inference_norm_layer_call_fn_10873562MЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
'__inference_norm_layer_call_fn_10873627i=Ђ:
3Ђ0
*'
inputsџџџџџџџџџа
p
Њ ""џџџџџџџџџа
'__inference_norm_layer_call_fn_10873636i=Ђ:
3Ђ0
*'
inputsџџџџџџџџџа
p 
Њ ""џџџџџџџџџаЄ
D__inference_output_layer_call_and_return_conditional_losses_10873676\34/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 |
)__inference_output_layer_call_fn_10873683O34/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЛ
&__inference_signature_wrapper_10873308#$-.34MЂJ
Ђ 
CЊ@
>
input_input/,
input_inputџџџџџџџџџв "/Њ,
*
output 
outputџџџџџџџџџ