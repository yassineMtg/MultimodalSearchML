ȟ
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.12v2.15.0-11-g63f5a65c7cd8�
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
Adam/v/dense_38/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_38/bias/*
dtype0*
shape:*%
shared_nameAdam/v/dense_38/bias
y
(Adam/v/dense_38/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_38/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_38/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_38/bias/*
dtype0*
shape:*%
shared_nameAdam/m/dense_38/bias
y
(Adam/m/dense_38/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_38/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_38/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_38/kernel/*
dtype0*
shape:	�*'
shared_nameAdam/v/dense_38/kernel
�
*Adam/v/dense_38/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_38/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_38/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_38/kernel/*
dtype0*
shape:	�*'
shared_nameAdam/m/dense_38/kernel
�
*Adam/m/dense_38/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_38/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense_37/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_37/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_37/bias
z
(Adam/v/dense_37/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_37/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_37/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_37/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_37/bias
z
(Adam/m/dense_37/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_37/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_37/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_37/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_37/kernel
�
*Adam/v/dense_37/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_37/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_37/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_37/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_37/kernel
�
*Adam/m/dense_37/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_37/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/dense_36/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_36/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_36/bias
z
(Adam/v/dense_36/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_36/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_36/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_36/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_36/bias
z
(Adam/m/dense_36/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_36/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_36/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_36/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_36/kernel
�
*Adam/v/dense_36/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_36/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_36/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_36/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_36/kernel
�
*Adam/m/dense_36/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_36/kernel* 
_output_shapes
:
��*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_38/biasVarHandleOp*
_output_shapes
: *

debug_namedense_38/bias/*
dtype0*
shape:*
shared_namedense_38/bias
k
!dense_38/bias/Read/ReadVariableOpReadVariableOpdense_38/bias*
_output_shapes
:*
dtype0
�
dense_38/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_38/kernel/*
dtype0*
shape:	�* 
shared_namedense_38/kernel
t
#dense_38/kernel/Read/ReadVariableOpReadVariableOpdense_38/kernel*
_output_shapes
:	�*
dtype0
�
dense_37/biasVarHandleOp*
_output_shapes
: *

debug_namedense_37/bias/*
dtype0*
shape:�*
shared_namedense_37/bias
l
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes	
:�*
dtype0
�
dense_37/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_37/kernel/*
dtype0*
shape:
��* 
shared_namedense_37/kernel
u
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel* 
_output_shapes
:
��*
dtype0
�
dense_36/biasVarHandleOp*
_output_shapes
: *

debug_namedense_36/bias/*
dtype0*
shape:�*
shared_namedense_36/bias
l
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes	
:�*
dtype0
�
dense_36/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_36/kernel/*
dtype0*
shape:
��* 
shared_namedense_36/kernel
u
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel* 
_output_shapes
:
��*
dtype0
�
serving_default_dense_36_inputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_36_inputdense_36/kerneldense_36/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_311355

NoOpNoOp
�/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�.
value�.B�. B�.
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias*
.
0
1
#2
$3
+4
,5*
.
0
1
#2
$3
+4
,5*
* 
�
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*

2trace_0
3trace_1* 

4trace_0
5trace_1* 
* 
�
6
_variables
7_iterations
8_learning_rate
9_index_dict
:
_momentums
;_velocities
<_update_step_xla*

=serving_default* 

0
1*

0
1*
* 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 
_Y
VARIABLE_VALUEdense_36/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_36/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Jtrace_0
Ktrace_1* 

Ltrace_0
Mtrace_1* 
* 

#0
$1*

#0
$1*
* 
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

Strace_0* 

Ttrace_0* 
_Y
VARIABLE_VALUEdense_37/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_37/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*
* 
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Ztrace_0* 

[trace_0* 
_Y
VARIABLE_VALUEdense_38/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_38/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

\0
]1*
* 
* 
* 
* 
* 
* 
b
70
^1
_2
`3
a4
b5
c6
d7
e8
f9
g10
h11
i12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
^0
`1
b2
d3
f4
h5*
.
_0
a1
c2
e3
g4
i5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
j	variables
k	keras_api
	ltotal
	mcount*
H
n	variables
o	keras_api
	ptotal
	qcount
r
_fn_kwargs*
a[
VARIABLE_VALUEAdam/m/dense_36/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_36/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_36/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_36/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_37/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_37/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_37/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_37/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_38/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_38/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_38/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_38/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*

l0
m1*

j	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

p0
q1*

n	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_36/kerneldense_36/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/bias	iterationlearning_rateAdam/m/dense_36/kernelAdam/v/dense_36/kernelAdam/m/dense_36/biasAdam/v/dense_36/biasAdam/m/dense_37/kernelAdam/v/dense_37/kernelAdam/m/dense_37/biasAdam/v/dense_37/biasAdam/m/dense_38/kernelAdam/v/dense_38/kernelAdam/m/dense_38/biasAdam/v/dense_38/biastotal_1count_1totalcountConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_311608
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_36/kerneldense_36/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/bias	iterationlearning_rateAdam/m/dense_36/kernelAdam/v/dense_36/kernelAdam/m/dense_36/biasAdam/v/dense_36/biasAdam/m/dense_37/kernelAdam/v/dense_37/kernelAdam/m/dense_37/biasAdam/v/dense_37/biasAdam/m/dense_38/kernelAdam/v/dense_38/kernelAdam/m/dense_38/biasAdam/v/dense_38/biastotal_1count_1totalcount*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_311689��
�
�
__inference__traced_save_311608
file_prefix:
&read_disablecopyonread_dense_36_kernel:
��5
&read_1_disablecopyonread_dense_36_bias:	�<
(read_2_disablecopyonread_dense_37_kernel:
��5
&read_3_disablecopyonread_dense_37_bias:	�;
(read_4_disablecopyonread_dense_38_kernel:	�4
&read_5_disablecopyonread_dense_38_bias:,
"read_6_disablecopyonread_iteration:	 0
&read_7_disablecopyonread_learning_rate: C
/read_8_disablecopyonread_adam_m_dense_36_kernel:
��C
/read_9_disablecopyonread_adam_v_dense_36_kernel:
��=
.read_10_disablecopyonread_adam_m_dense_36_bias:	�=
.read_11_disablecopyonread_adam_v_dense_36_bias:	�D
0read_12_disablecopyonread_adam_m_dense_37_kernel:
��D
0read_13_disablecopyonread_adam_v_dense_37_kernel:
��=
.read_14_disablecopyonread_adam_m_dense_37_bias:	�=
.read_15_disablecopyonread_adam_v_dense_37_bias:	�C
0read_16_disablecopyonread_adam_m_dense_38_kernel:	�C
0read_17_disablecopyonread_adam_v_dense_38_kernel:	�<
.read_18_disablecopyonread_adam_m_dense_38_bias:<
.read_19_disablecopyonread_adam_v_dense_38_bias:+
!read_20_disablecopyonread_total_1: +
!read_21_disablecopyonread_count_1: )
read_22_disablecopyonread_total: )
read_23_disablecopyonread_count: 
savev2_const
identity_49��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: x
Read/DisableCopyOnReadDisableCopyOnRead&read_disablecopyonread_dense_36_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp&read_disablecopyonread_dense_36_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_dense_36_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_dense_36_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_2/DisableCopyOnReadDisableCopyOnRead(read_2_disablecopyonread_dense_37_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp(read_2_disablecopyonread_dense_37_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��z
Read_3/DisableCopyOnReadDisableCopyOnRead&read_3_disablecopyonread_dense_37_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp&read_3_disablecopyonread_dense_37_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_38_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_38_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	�z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_38_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_38_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_iteration^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_learning_rate^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead/read_8_disablecopyonread_adam_m_dense_36_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp/read_8_disablecopyonread_adam_m_dense_36_kernel^Read_8/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_9/DisableCopyOnReadDisableCopyOnRead/read_9_disablecopyonread_adam_v_dense_36_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp/read_9_disablecopyonread_adam_v_dense_36_kernel^Read_9/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_10/DisableCopyOnReadDisableCopyOnRead.read_10_disablecopyonread_adam_m_dense_36_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp.read_10_disablecopyonread_adam_m_dense_36_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_11/DisableCopyOnReadDisableCopyOnRead.read_11_disablecopyonread_adam_v_dense_36_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp.read_11_disablecopyonread_adam_v_dense_36_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_12/DisableCopyOnReadDisableCopyOnRead0read_12_disablecopyonread_adam_m_dense_37_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp0read_12_disablecopyonread_adam_m_dense_37_kernel^Read_12/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_13/DisableCopyOnReadDisableCopyOnRead0read_13_disablecopyonread_adam_v_dense_37_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp0read_13_disablecopyonread_adam_v_dense_37_kernel^Read_13/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_14/DisableCopyOnReadDisableCopyOnRead.read_14_disablecopyonread_adam_m_dense_37_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp.read_14_disablecopyonread_adam_m_dense_37_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_15/DisableCopyOnReadDisableCopyOnRead.read_15_disablecopyonread_adam_v_dense_37_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp.read_15_disablecopyonread_adam_v_dense_37_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnRead0read_16_disablecopyonread_adam_m_dense_38_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp0read_16_disablecopyonread_adam_m_dense_38_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_adam_v_dense_38_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_adam_v_dense_38_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_18/DisableCopyOnReadDisableCopyOnRead.read_18_disablecopyonread_adam_m_dense_38_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp.read_18_disablecopyonread_adam_m_dense_38_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead.read_19_disablecopyonread_adam_v_dense_38_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp.read_19_disablecopyonread_adam_v_dense_38_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_20/DisableCopyOnReadDisableCopyOnRead!read_20_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp!read_20_disablecopyonread_total_1^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_21/DisableCopyOnReadDisableCopyOnRead!read_21_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp!read_21_disablecopyonread_count_1^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_22/DisableCopyOnReadDisableCopyOnReadread_22_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpread_22_disablecopyonread_total^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_23/DisableCopyOnReadDisableCopyOnReadread_23_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOpread_23_disablecopyonread_count^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_48Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_49IdentityIdentity_48:output:0^NoOp*
T0*
_output_shapes
: �

NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_49Identity_49:output:0*(
_construction_contextkEagerRuntime*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_namedense_36/kernel:-)
'
_user_specified_namedense_36/bias:/+
)
_user_specified_namedense_37/kernel:-)
'
_user_specified_namedense_37/bias:/+
)
_user_specified_namedense_38/kernel:-)
'
_user_specified_namedense_38/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:6	2
0
_user_specified_nameAdam/m/dense_36/kernel:6
2
0
_user_specified_nameAdam/v/dense_36/kernel:40
.
_user_specified_nameAdam/m/dense_36/bias:40
.
_user_specified_nameAdam/v/dense_36/bias:62
0
_user_specified_nameAdam/m/dense_37/kernel:62
0
_user_specified_nameAdam/v/dense_37/kernel:40
.
_user_specified_nameAdam/m/dense_37/bias:40
.
_user_specified_nameAdam/v/dense_37/bias:62
0
_user_specified_nameAdam/m/dense_38/kernel:62
0
_user_specified_nameAdam/v/dense_38/kernel:40
.
_user_specified_nameAdam/m/dense_38/bias:40
.
_user_specified_nameAdam/v/dense_38/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount:=9

_output_shapes
: 

_user_specified_nameConst
�

�
D__inference_dense_36_layer_call_and_return_conditional_losses_311375

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_311402

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_dense_36_layer_call_fn_311364

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_311187p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name311358:&"
 
_user_specified_name311360
�

�
.__inference_sequential_12_layer_call_fn_311298
dense_36_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_311264o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_36_input:&"
 
_user_specified_name311284:&"
 
_user_specified_name311286:&"
 
_user_specified_name311288:&"
 
_user_specified_name311290:&"
 
_user_specified_name311292:&"
 
_user_specified_name311294
�

�
D__inference_dense_37_layer_call_and_return_conditional_losses_311422

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
D__inference_dense_37_layer_call_and_return_conditional_losses_311216

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
G
+__inference_dropout_12_layer_call_fn_311385

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_311251a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
D__inference_dense_38_layer_call_and_return_conditional_losses_311442

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�p
�
"__inference__traced_restore_311689
file_prefix4
 assignvariableop_dense_36_kernel:
��/
 assignvariableop_1_dense_36_bias:	�6
"assignvariableop_2_dense_37_kernel:
��/
 assignvariableop_3_dense_37_bias:	�5
"assignvariableop_4_dense_38_kernel:	�.
 assignvariableop_5_dense_38_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: =
)assignvariableop_8_adam_m_dense_36_kernel:
��=
)assignvariableop_9_adam_v_dense_36_kernel:
��7
(assignvariableop_10_adam_m_dense_36_bias:	�7
(assignvariableop_11_adam_v_dense_36_bias:	�>
*assignvariableop_12_adam_m_dense_37_kernel:
��>
*assignvariableop_13_adam_v_dense_37_kernel:
��7
(assignvariableop_14_adam_m_dense_37_bias:	�7
(assignvariableop_15_adam_v_dense_37_bias:	�=
*assignvariableop_16_adam_m_dense_38_kernel:	�=
*assignvariableop_17_adam_v_dense_38_kernel:	�6
(assignvariableop_18_adam_m_dense_38_bias:6
(assignvariableop_19_adam_v_dense_38_bias:%
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: 
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_36_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_36_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_37_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_37_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_38_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_38_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp)assignvariableop_8_adam_m_dense_36_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp)assignvariableop_9_adam_v_dense_36_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp(assignvariableop_10_adam_m_dense_36_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp(assignvariableop_11_adam_v_dense_36_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp*assignvariableop_12_adam_m_dense_37_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_v_dense_37_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_m_dense_37_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_v_dense_37_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_m_dense_38_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_v_dense_38_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_m_dense_38_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_v_dense_38_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_25Identity_25:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:/+
)
_user_specified_namedense_36/kernel:-)
'
_user_specified_namedense_36/bias:/+
)
_user_specified_namedense_37/kernel:-)
'
_user_specified_namedense_37/bias:/+
)
_user_specified_namedense_38/kernel:-)
'
_user_specified_namedense_38/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:6	2
0
_user_specified_nameAdam/m/dense_36/kernel:6
2
0
_user_specified_nameAdam/v/dense_36/kernel:40
.
_user_specified_nameAdam/m/dense_36/bias:40
.
_user_specified_nameAdam/v/dense_36/bias:62
0
_user_specified_nameAdam/m/dense_37/kernel:62
0
_user_specified_nameAdam/v/dense_37/kernel:40
.
_user_specified_nameAdam/m/dense_37/bias:40
.
_user_specified_nameAdam/v/dense_37/bias:62
0
_user_specified_nameAdam/m/dense_38/kernel:62
0
_user_specified_nameAdam/v/dense_38/kernel:40
.
_user_specified_nameAdam/m/dense_38/bias:40
.
_user_specified_nameAdam/v/dense_38/bias:'#
!
_user_specified_name	total_1:'#
!
_user_specified_name	count_1:%!

_user_specified_nametotal:%!

_user_specified_namecount
�
�
I__inference_sequential_12_layer_call_and_return_conditional_losses_311264
dense_36_input#
dense_36_311242:
��
dense_36_311244:	�#
dense_37_311253:
��
dense_37_311255:	�"
dense_38_311258:	�
dense_38_311260:
identity�� dense_36/StatefulPartitionedCall� dense_37/StatefulPartitionedCall� dense_38/StatefulPartitionedCall�
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_311242dense_36_311244*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_311187�
dropout_12/PartitionedCallPartitionedCall)dense_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_311251�
 dense_37/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0dense_37_311253dense_37_311255*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_311216�
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_311258dense_38_311260*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_311232x
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_36_input:&"
 
_user_specified_name311242:&"
 
_user_specified_name311244:&"
 
_user_specified_name311253:&"
 
_user_specified_name311255:&"
 
_user_specified_name311258:&"
 
_user_specified_name311260
�
d
F__inference_dropout_12_layer_call_and_return_conditional_losses_311251

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
.__inference_sequential_12_layer_call_fn_311281
dense_36_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_311239o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_36_input:&"
 
_user_specified_name311267:&"
 
_user_specified_name311269:&"
 
_user_specified_name311271:&"
 
_user_specified_name311273:&"
 
_user_specified_name311275:&"
 
_user_specified_name311277
�

�
$__inference_signature_wrapper_311355
dense_36_input
unknown:
��
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_311174o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_36_input:&"
 
_user_specified_name311341:&"
 
_user_specified_name311343:&"
 
_user_specified_name311345:&"
 
_user_specified_name311347:&"
 
_user_specified_name311349:&"
 
_user_specified_name311351
�
�
)__inference_dense_38_layer_call_fn_311431

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_311232o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name311425:&"
 
_user_specified_name311427
�

�
D__inference_dense_38_layer_call_and_return_conditional_losses_311232

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
D__inference_dense_36_layer_call_and_return_conditional_losses_311187

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
d
+__inference_dropout_12_layer_call_fn_311380

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_311204p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
I__inference_sequential_12_layer_call_and_return_conditional_losses_311239
dense_36_input#
dense_36_311188:
��
dense_36_311190:	�#
dense_37_311217:
��
dense_37_311219:	�"
dense_38_311233:	�
dense_38_311235:
identity�� dense_36/StatefulPartitionedCall� dense_37/StatefulPartitionedCall� dense_38/StatefulPartitionedCall�"dropout_12/StatefulPartitionedCall�
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_311188dense_36_311190*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_311187�
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_12_layer_call_and_return_conditional_losses_311204�
 dense_37/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0dense_37_311217dense_37_311219*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_311216�
 dense_38/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0dense_38_311233dense_38_311235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_38_layer_call_and_return_conditional_losses_311232x
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall:X T
(
_output_shapes
:����������
(
_user_specified_namedense_36_input:&"
 
_user_specified_name311188:&"
 
_user_specified_name311190:&"
 
_user_specified_name311217:&"
 
_user_specified_name311219:&"
 
_user_specified_name311233:&"
 
_user_specified_name311235
�&
�
!__inference__wrapped_model_311174
dense_36_inputI
5sequential_12_dense_36_matmul_readvariableop_resource:
��E
6sequential_12_dense_36_biasadd_readvariableop_resource:	�I
5sequential_12_dense_37_matmul_readvariableop_resource:
��E
6sequential_12_dense_37_biasadd_readvariableop_resource:	�H
5sequential_12_dense_38_matmul_readvariableop_resource:	�D
6sequential_12_dense_38_biasadd_readvariableop_resource:
identity��-sequential_12/dense_36/BiasAdd/ReadVariableOp�,sequential_12/dense_36/MatMul/ReadVariableOp�-sequential_12/dense_37/BiasAdd/ReadVariableOp�,sequential_12/dense_37/MatMul/ReadVariableOp�-sequential_12/dense_38/BiasAdd/ReadVariableOp�,sequential_12/dense_38/MatMul/ReadVariableOp�
,sequential_12/dense_36/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_36_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_12/dense_36/MatMulMatMuldense_36_input4sequential_12/dense_36/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_12/dense_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_36_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_12/dense_36/BiasAddBiasAdd'sequential_12/dense_36/MatMul:product:05sequential_12/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_12/dense_36/ReluRelu'sequential_12/dense_36/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!sequential_12/dropout_12/IdentityIdentity)sequential_12/dense_36/Relu:activations:0*
T0*(
_output_shapes
:�����������
,sequential_12/dense_37/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_37_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_12/dense_37/MatMulMatMul*sequential_12/dropout_12/Identity:output:04sequential_12/dense_37/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_12/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_37_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_12/dense_37/BiasAddBiasAdd'sequential_12/dense_37/MatMul:product:05sequential_12/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_12/dense_37/ReluRelu'sequential_12/dense_37/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,sequential_12/dense_38/MatMul/ReadVariableOpReadVariableOp5sequential_12_dense_38_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_12/dense_38/MatMulMatMul)sequential_12/dense_37/Relu:activations:04sequential_12/dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-sequential_12/dense_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_12/dense_38/BiasAddBiasAdd'sequential_12/dense_38/MatMul:product:05sequential_12/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential_12/dense_38/SigmoidSigmoid'sequential_12/dense_38/BiasAdd:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"sequential_12/dense_38/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^sequential_12/dense_36/BiasAdd/ReadVariableOp-^sequential_12/dense_36/MatMul/ReadVariableOp.^sequential_12/dense_37/BiasAdd/ReadVariableOp-^sequential_12/dense_37/MatMul/ReadVariableOp.^sequential_12/dense_38/BiasAdd/ReadVariableOp-^sequential_12/dense_38/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2^
-sequential_12/dense_36/BiasAdd/ReadVariableOp-sequential_12/dense_36/BiasAdd/ReadVariableOp2\
,sequential_12/dense_36/MatMul/ReadVariableOp,sequential_12/dense_36/MatMul/ReadVariableOp2^
-sequential_12/dense_37/BiasAdd/ReadVariableOp-sequential_12/dense_37/BiasAdd/ReadVariableOp2\
,sequential_12/dense_37/MatMul/ReadVariableOp,sequential_12/dense_37/MatMul/ReadVariableOp2^
-sequential_12/dense_38/BiasAdd/ReadVariableOp-sequential_12/dense_38/BiasAdd/ReadVariableOp2\
,sequential_12/dense_38/MatMul/ReadVariableOp,sequential_12/dense_38/MatMul/ReadVariableOp:X T
(
_output_shapes
:����������
(
_user_specified_namedense_36_input:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
)__inference_dense_37_layer_call_fn_311411

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_311216p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name311405:&"
 
_user_specified_name311407
�

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_311397

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_12_layer_call_and_return_conditional_losses_311204

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
J
dense_36_input8
 serving_default_dense_36_input:0����������<
dense_380
StatefulPartitionedCall:0���������tensorflow/serving/predict:�y
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias"
_tf_keras_layer
J
0
1
#2
$3
+4
,5"
trackable_list_wrapper
J
0
1
#2
$3
+4
,5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
2trace_0
3trace_12�
.__inference_sequential_12_layer_call_fn_311281
.__inference_sequential_12_layer_call_fn_311298�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z2trace_0z3trace_1
�
4trace_0
5trace_12�
I__inference_sequential_12_layer_call_and_return_conditional_losses_311239
I__inference_sequential_12_layer_call_and_return_conditional_losses_311264�
���
FullArgSpec)
args!�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z4trace_0z5trace_1
�B�
!__inference__wrapped_model_311174dense_36_input"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
6
_variables
7_iterations
8_learning_rate
9_index_dict
:
_momentums
;_velocities
<_update_step_xla"
experimentalOptimizer
,
=serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ctrace_02�
)__inference_dense_36_layer_call_fn_311364�
���
FullArgSpec
args�

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
annotations� *
 zCtrace_0
�
Dtrace_02�
D__inference_dense_36_layer_call_and_return_conditional_losses_311375�
���
FullArgSpec
args�

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
annotations� *
 zDtrace_0
#:!
��2dense_36/kernel
:�2dense_36/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Jtrace_0
Ktrace_12�
+__inference_dropout_12_layer_call_fn_311380
+__inference_dropout_12_layer_call_fn_311385�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0zKtrace_1
�
Ltrace_0
Mtrace_12�
F__inference_dropout_12_layer_call_and_return_conditional_losses_311397
F__inference_dropout_12_layer_call_and_return_conditional_losses_311402�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0zMtrace_1
"
_generic_user_object
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
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
Strace_02�
)__inference_dense_37_layer_call_fn_311411�
���
FullArgSpec
args�

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
annotations� *
 zStrace_0
�
Ttrace_02�
D__inference_dense_37_layer_call_and_return_conditional_losses_311422�
���
FullArgSpec
args�

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
annotations� *
 zTtrace_0
#:!
��2dense_37/kernel
:�2dense_37/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
Ztrace_02�
)__inference_dense_38_layer_call_fn_311431�
���
FullArgSpec
args�

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
annotations� *
 zZtrace_0
�
[trace_02�
D__inference_dense_38_layer_call_and_return_conditional_losses_311442�
���
FullArgSpec
args�

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
annotations� *
 z[trace_0
": 	�2dense_38/kernel
:2dense_38/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_12_layer_call_fn_311281dense_36_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_12_layer_call_fn_311298dense_36_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_12_layer_call_and_return_conditional_losses_311239dense_36_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_12_layer_call_and_return_conditional_losses_311264dense_36_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
~
70
^1
_2
`3
a4
b5
c6
d7
e8
f9
g10
h11
i12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
^0
`1
b2
d3
f4
h5"
trackable_list_wrapper
J
_0
a1
c2
e3
g4
i5"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_311355dense_36_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 #

kwonlyargs�
jdense_36_input
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_36_layer_call_fn_311364inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
D__inference_dense_36_layer_call_and_return_conditional_losses_311375inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_12_layer_call_fn_311380inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_12_layer_call_fn_311385inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_12_layer_call_and_return_conditional_losses_311397inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_12_layer_call_and_return_conditional_losses_311402inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_37_layer_call_fn_311411inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
D__inference_dense_37_layer_call_and_return_conditional_losses_311422inputs"�
���
FullArgSpec
args�

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
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_38_layer_call_fn_311431inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
D__inference_dense_38_layer_call_and_return_conditional_losses_311442inputs"�
���
FullArgSpec
args�

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
annotations� *
 
N
j	variables
k	keras_api
	ltotal
	mcount"
_tf_keras_metric
^
n	variables
o	keras_api
	ptotal
	qcount
r
_fn_kwargs"
_tf_keras_metric
(:&
��2Adam/m/dense_36/kernel
(:&
��2Adam/v/dense_36/kernel
!:�2Adam/m/dense_36/bias
!:�2Adam/v/dense_36/bias
(:&
��2Adam/m/dense_37/kernel
(:&
��2Adam/v/dense_37/kernel
!:�2Adam/m/dense_37/bias
!:�2Adam/v/dense_37/bias
':%	�2Adam/m/dense_38/kernel
':%	�2Adam/v/dense_38/kernel
 :2Adam/m/dense_38/bias
 :2Adam/v/dense_38/bias
.
l0
m1"
trackable_list_wrapper
-
j	variables"
_generic_user_object
:  (2total
:  (2count
.
p0
q1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
!__inference__wrapped_model_311174w#$+,8�5
.�+
)�&
dense_36_input����������
� "3�0
.
dense_38"�
dense_38����������
D__inference_dense_36_layer_call_and_return_conditional_losses_311375e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_36_layer_call_fn_311364Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
D__inference_dense_37_layer_call_and_return_conditional_losses_311422e#$0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_37_layer_call_fn_311411Z#$0�-
&�#
!�
inputs����������
� ""�
unknown�����������
D__inference_dense_38_layer_call_and_return_conditional_losses_311442d+,0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
)__inference_dense_38_layer_call_fn_311431Y+,0�-
&�#
!�
inputs����������
� "!�
unknown����������
F__inference_dropout_12_layer_call_and_return_conditional_losses_311397e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
F__inference_dropout_12_layer_call_and_return_conditional_losses_311402e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
+__inference_dropout_12_layer_call_fn_311380Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
+__inference_dropout_12_layer_call_fn_311385Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
I__inference_sequential_12_layer_call_and_return_conditional_losses_311239x#$+,@�=
6�3
)�&
dense_36_input����������
p

 
� ",�)
"�
tensor_0���������
� �
I__inference_sequential_12_layer_call_and_return_conditional_losses_311264x#$+,@�=
6�3
)�&
dense_36_input����������
p 

 
� ",�)
"�
tensor_0���������
� �
.__inference_sequential_12_layer_call_fn_311281m#$+,@�=
6�3
)�&
dense_36_input����������
p

 
� "!�
unknown����������
.__inference_sequential_12_layer_call_fn_311298m#$+,@�=
6�3
)�&
dense_36_input����������
p 

 
� "!�
unknown����������
$__inference_signature_wrapper_311355�#$+,J�G
� 
@�=
;
dense_36_input)�&
dense_36_input����������"3�0
.
dense_38"�
dense_38���������