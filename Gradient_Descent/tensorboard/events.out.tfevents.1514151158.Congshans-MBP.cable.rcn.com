       ЃK"	  =жAbrain.Event:2TїлAјѓ      IъPЈ	J%Љ=жA"ыч
W
keras_learning_phasePlaceholder*
_output_shapes
:*
shape: *
dtype0

Y
input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
shape: *
dtype0
o
dense_1/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
b
dense_1/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
d
dense_1/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ЋвШ>
А
(dense_1/truncated_normal/TruncatedNormalTruncatedNormaldense_1/truncated_normal/shape*
seedБџх)*
_output_shapes

:*
seed2юН*
dtype0*
T0

dense_1/truncated_normal/mulMul(dense_1/truncated_normal/TruncatedNormaldense_1/truncated_normal/stddev*
_output_shapes

:*
T0

dense_1/truncated_normalAdddense_1/truncated_normal/muldense_1/truncated_normal/mean*
_output_shapes

:*
T0

dense_1/kernel
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
О
dense_1/kernel/AssignAssigndense_1/kerneldense_1/truncated_normal*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel*
_output_shapes

:*
T0
{
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes

:*
T0
Z
dense_1/ConstConst*
_output_shapes
:*
dtype0*
valueB*    
x
dense_1/bias
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
Љ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias*
_output_shapes
:*
T0
q
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes
:*
T0

dense_1/MatMulMatMulinput_1dense_1/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
transpose_a( *
T0

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC*
T0
W
dense_1/TanhTanhdense_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
o
dense_2/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
b
dense_2/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
d
dense_2/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
А
(dense_2/truncated_normal/TruncatedNormalTruncatedNormaldense_2/truncated_normal/shape*
seedБџх)*
_output_shapes

:*
seed2ў*
dtype0*
T0

dense_2/truncated_normal/mulMul(dense_2/truncated_normal/TruncatedNormaldense_2/truncated_normal/stddev*
_output_shapes

:*
T0

dense_2/truncated_normalAdddense_2/truncated_normal/muldense_2/truncated_normal/mean*
_output_shapes

:*
T0

dense_2/kernel
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
О
dense_2/kernel/AssignAssigndense_2/kerneldense_2/truncated_normal*
use_locking(*
validate_shape(*!
_class
loc:@dense_2/kernel*
_output_shapes

:*
T0
{
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
_output_shapes

:*
T0
Z
dense_2/ConstConst*
_output_shapes
:*
dtype0*
valueB*    
x
dense_2/bias
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
Љ
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
validate_shape(*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0
q
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0

dense_2/MatMulMatMuldense_1/Tanhdense_2/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
transpose_a( *
T0

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC*
T0
W
dense_2/TanhTanhdense_2/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
o
dense_3/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
b
dense_3/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
d
dense_3/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ш!?
Џ
(dense_3/truncated_normal/TruncatedNormalTruncatedNormaldense_3/truncated_normal/shape*
seedБџх)*
_output_shapes

:*
seed2щX*
dtype0*
T0

dense_3/truncated_normal/mulMul(dense_3/truncated_normal/TruncatedNormaldense_3/truncated_normal/stddev*
_output_shapes

:*
T0

dense_3/truncated_normalAdddense_3/truncated_normal/muldense_3/truncated_normal/mean*
_output_shapes

:*
T0

dense_3/kernel
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
О
dense_3/kernel/AssignAssigndense_3/kerneldense_3/truncated_normal*
use_locking(*
validate_shape(*!
_class
loc:@dense_3/kernel*
_output_shapes

:*
T0
{
dense_3/kernel/readIdentitydense_3/kernel*!
_class
loc:@dense_3/kernel*
_output_shapes

:*
T0
Z
dense_3/ConstConst*
_output_shapes
:*
dtype0*
valueB*    
x
dense_3/bias
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
Љ
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
use_locking(*
validate_shape(*
_class
loc:@dense_3/bias*
_output_shapes
:*
T0
q
dense_3/bias/readIdentitydense_3/bias*
_class
loc:@dense_3/bias*
_output_shapes
:*
T0

dense_3/MatMulMatMuldense_2/Tanhdense_3/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
transpose_a( *
T0

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC*
T0
W
dense_3/TanhTanhdense_3/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
m
dense_4/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
_
dense_4/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *7П
_
dense_4/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *7?
Ї
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
seedБџх)*
_output_shapes

:*
seed2їЛ9*
dtype0*
T0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0

dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
_output_shapes

:*
T0
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
_output_shapes

:*
T0

dense_4/kernel
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
М
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
validate_shape(*!
_class
loc:@dense_4/kernel*
_output_shapes

:*
T0
{
dense_4/kernel/readIdentitydense_4/kernel*!
_class
loc:@dense_4/kernel*
_output_shapes

:*
T0
Z
dense_4/ConstConst*
_output_shapes
:*
dtype0*
valueB*    
x
dense_4/bias
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
Љ
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
validate_shape(*
_class
loc:@dense_4/bias*
_output_shapes
:*
T0
q
dense_4/bias/readIdentitydense_4/bias*
_class
loc:@dense_4/bias*
_output_shapes
:*
T0

dense_4/MatMulMatMuldense_3/Tanhdense_4/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_b( *
transpose_a( *
T0

dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC*
T0
]
dense_4/SoftmaxSoftmaxdense_4/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
U
lr/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL=
f
lr
VariableV2*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 

	lr/AssignAssignlrlr/initial_value*
use_locking(*
validate_shape(*
_class
	loc:@lr*
_output_shapes
: *
T0
O
lr/readIdentitylr*
_class
	loc:@lr*
_output_shapes
: *
T0
V
rho/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *fff?
g
rho
VariableV2*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 


rho/AssignAssignrhorho/initial_value*
use_locking(*
validate_shape(*
_class

loc:@rho*
_output_shapes
: *
T0
R
rho/readIdentityrho*
_class

loc:@rho*
_output_shapes
: *
T0
X
decay/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
i
decay
VariableV2*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 

decay/AssignAssigndecaydecay/initial_value*
use_locking(*
validate_shape(*
_class

loc:@decay*
_output_shapes
: *
T0
X

decay/readIdentitydecay*
_class

loc:@decay*
_output_shapes
: *
T0
]
iterations/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
n

iterations
VariableV2*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 
Њ
iterations/AssignAssign
iterationsiterations/initial_value*
use_locking(*
validate_shape(*
_class
loc:@iterations*
_output_shapes
: *
T0
g
iterations/readIdentity
iterations*
_class
loc:@iterations*
_output_shapes
: *
T0
d
dense_4_sample_weightsPlaceholder*#
_output_shapes
:џџџџџџџџџ*
shape: *
dtype0
i
dense_4_targetPlaceholder*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shape: *
dtype0
W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :

SumSumdense_4/SoftmaxSum/reduction_indices*
	keep_dims(*'
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
Z
truedivRealDivdense_4/SoftmaxSum*'
_output_shapes
:џџџџџџџџџ*
T0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
9
subSubsub/xConst*
_output_shapes
: *
T0
`
clip_by_value/MinimumMinimumtruedivsub*'
_output_shapes
:џџџџџџџџџ*
T0
h
clip_by_valueMaximumclip_by_value/MinimumConst*'
_output_shapes
:џџџџџџџџџ*
T0
K
LogLogclip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
Q
mulMuldense_4_targetLog*'
_output_shapes
:џџџџџџџџџ*
T0
Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
u
Sum_1SummulSum_1/reduction_indices*
	keep_dims( *#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
?
NegNegSum_1*#
_output_shapes
:џџџџџџџџџ*
T0
Y
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB 
t
MeanMeanNegMean/reduction_indices*
	keep_dims( *#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
X
mul_1MulMeandense_4_sample_weights*#
_output_shapes
:џџџџџџџџџ*
T0
O

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
f
NotEqualNotEqualdense_4_sample_weights
NotEqual/y*#
_output_shapes
:џџџџџџџџџ*
T0
S
CastCastNotEqual*

DstT0*

SrcT0
*#
_output_shapes
:џџџџџџџџџ
Q
Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
[
Mean_1MeanCastConst_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
Q
	truediv_1RealDivmul_1Mean_1*#
_output_shapes
:џџџџџџџџџ*
T0
Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
`
Mean_2Mean	truediv_1Const_2*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
>
mul_2Mulmul_2/xMean_2*
_output_shapes
: *
T0
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
l
ArgMaxArgMaxdense_4_targetArgMax/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
q
ArgMax_1ArgMaxdense_4/SoftmaxArgMax_1/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:џџџџџџџџџ*
T0	
R
Cast_1CastEqual*

DstT0*

SrcT0
*#
_output_shapes
:џџџџџџџџџ
Q
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
]
Mean_3MeanCast_1Const_3*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
#

group_depsNoOp^mul_2^Mean_3
l
gradients/ShapeConst*
_class

loc:@mul_2*
_output_shapes
: *
dtype0*
valueB 
n
gradients/ConstConst*
_class

loc:@mul_2*
_output_shapes
: *
dtype0*
valueB
 *  ?
s
gradients/FillFillgradients/Shapegradients/Const*
_class

loc:@mul_2*
_output_shapes
: *
T0
w
gradients/mul_2_grad/ShapeConst*
_class

loc:@mul_2*
_output_shapes
: *
dtype0*
valueB 
y
gradients/mul_2_grad/Shape_1Const*
_class

loc:@mul_2*
_output_shapes
: *
dtype0*
valueB 
д
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
_class

loc:@mul_2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
r
gradients/mul_2_grad/mulMulgradients/FillMean_2*
_class

loc:@mul_2*
_output_shapes
: *
T0
П
gradients/mul_2_grad/SumSumgradients/mul_2_grad/mul*gradients/mul_2_grad/BroadcastGradientArgs*
_class

loc:@mul_2*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
І
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
_class

loc:@mul_2*
_output_shapes
: *
Tshape0*
T0
u
gradients/mul_2_grad/mul_1Mulmul_2/xgradients/Fill*
_class

loc:@mul_2*
_output_shapes
: *
T0
Х
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
_class

loc:@mul_2*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Ќ
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
_class

loc:@mul_2*
_output_shapes
: *
Tshape0*
T0

#gradients/Mean_2_grad/Reshape/shapeConst*
_class
loc:@Mean_2*
_output_shapes
:*
dtype0*
valueB:
Л
gradients/Mean_2_grad/ReshapeReshapegradients/mul_2_grad/Reshape_1#gradients/Mean_2_grad/Reshape/shape*
_class
loc:@Mean_2*
_output_shapes
:*
Tshape0*
T0

gradients/Mean_2_grad/ShapeShape	truediv_1*
_class
loc:@Mean_2*
_output_shapes
:*
out_type0*
T0
Й
gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Shape*
_class
loc:@Mean_2*

Tmultiples0*#
_output_shapes
:џџџџџџџџџ*
T0

gradients/Mean_2_grad/Shape_1Shape	truediv_1*
_class
loc:@Mean_2*
_output_shapes
:*
out_type0*
T0
{
gradients/Mean_2_grad/Shape_2Const*
_class
loc:@Mean_2*
_output_shapes
: *
dtype0*
valueB 

gradients/Mean_2_grad/ConstConst*
_class
loc:@Mean_2*
_output_shapes
:*
dtype0*
valueB: 
З
gradients/Mean_2_grad/ProdProdgradients/Mean_2_grad/Shape_1gradients/Mean_2_grad/Const*
_class
loc:@Mean_2*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0

gradients/Mean_2_grad/Const_1Const*
_class
loc:@Mean_2*
_output_shapes
:*
dtype0*
valueB: 
Л
gradients/Mean_2_grad/Prod_1Prodgradients/Mean_2_grad/Shape_2gradients/Mean_2_grad/Const_1*
_class
loc:@Mean_2*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
|
gradients/Mean_2_grad/Maximum/yConst*
_class
loc:@Mean_2*
_output_shapes
: *
dtype0*
value	B :
Ѓ
gradients/Mean_2_grad/MaximumMaximumgradients/Mean_2_grad/Prod_1gradients/Mean_2_grad/Maximum/y*
_class
loc:@Mean_2*
_output_shapes
: *
T0
Ё
gradients/Mean_2_grad/floordivFloorDivgradients/Mean_2_grad/Prodgradients/Mean_2_grad/Maximum*
_class
loc:@Mean_2*
_output_shapes
: *
T0

gradients/Mean_2_grad/CastCastgradients/Mean_2_grad/floordiv*

DstT0*
_class
loc:@Mean_2*

SrcT0*
_output_shapes
: 
Љ
gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Cast*
_class
loc:@Mean_2*#
_output_shapes
:џџџџџџџџџ*
T0

gradients/truediv_1_grad/ShapeShapemul_1*
_class
loc:@truediv_1*
_output_shapes
:*
out_type0*
T0

 gradients/truediv_1_grad/Shape_1Const*
_class
loc:@truediv_1*
_output_shapes
: *
dtype0*
valueB 
ф
.gradients/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_1_grad/Shape gradients/truediv_1_grad/Shape_1*
_class
loc:@truediv_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

 gradients/truediv_1_grad/RealDivRealDivgradients/Mean_2_grad/truedivMean_1*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0
г
gradients/truediv_1_grad/SumSum gradients/truediv_1_grad/RealDiv.gradients/truediv_1_grad/BroadcastGradientArgs*
_class
loc:@truediv_1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
У
 gradients/truediv_1_grad/ReshapeReshapegradients/truediv_1_grad/Sumgradients/truediv_1_grad/Shape*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
v
gradients/truediv_1_grad/NegNegmul_1*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0

"gradients/truediv_1_grad/RealDiv_1RealDivgradients/truediv_1_grad/NegMean_1*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0
Ѕ
"gradients/truediv_1_grad/RealDiv_2RealDiv"gradients/truediv_1_grad/RealDiv_1Mean_1*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0
В
gradients/truediv_1_grad/mulMulgradients/Mean_2_grad/truediv"gradients/truediv_1_grad/RealDiv_2*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0
г
gradients/truediv_1_grad/Sum_1Sumgradients/truediv_1_grad/mul0gradients/truediv_1_grad/BroadcastGradientArgs:1*
_class
loc:@truediv_1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
М
"gradients/truediv_1_grad/Reshape_1Reshapegradients/truediv_1_grad/Sum_1 gradients/truediv_1_grad/Shape_1*
_class
loc:@truediv_1*
_output_shapes
: *
Tshape0*
T0
x
gradients/mul_1_grad/ShapeShapeMean*
_class

loc:@mul_1*
_output_shapes
:*
out_type0*
T0

gradients/mul_1_grad/Shape_1Shapedense_4_sample_weights*
_class

loc:@mul_1*
_output_shapes
:*
out_type0*
T0
д
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
_class

loc:@mul_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ё
gradients/mul_1_grad/mulMul gradients/truediv_1_grad/Reshapedense_4_sample_weights*
_class

loc:@mul_1*#
_output_shapes
:џџџџџџџџџ*
T0
П
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
_class

loc:@mul_1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Г
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
_class

loc:@mul_1*#
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0

gradients/mul_1_grad/mul_1MulMean gradients/truediv_1_grad/Reshape*
_class

loc:@mul_1*#
_output_shapes
:џџџџџџџџџ*
T0
Х
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
_class

loc:@mul_1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Й
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
_class

loc:@mul_1*#
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
u
gradients/Mean_grad/ShapeShapeNeg*
_class
	loc:@Mean*
_output_shapes
:*
out_type0*
T0
s
gradients/Mean_grad/SizeConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B :

gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
_class
	loc:@Mean*
_output_shapes
: *
T0

gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
_class
	loc:@Mean*
_output_shapes
: *
T0
~
gradients/Mean_grad/Shape_1Const*
_class
	loc:@Mean*
_output_shapes
:*
dtype0*
valueB: 
z
gradients/Mean_grad/range/startConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B : 
z
gradients/Mean_grad/range/deltaConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B :
П
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*
_class
	loc:@Mean*
_output_shapes
:*

Tidx0
y
gradients/Mean_grad/Fill/valueConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B :

gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
_class
	loc:@Mean*
_output_shapes
: *
T0
ы
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
_class
	loc:@Mean*#
_output_shapes
:џџџџџџџџџ*
N*
T0
x
gradients/Mean_grad/Maximum/yConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B :
Џ
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
_class
	loc:@Mean*#
_output_shapes
:џџџџџџџџџ*
T0
Ї
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
_class
	loc:@Mean*#
_output_shapes
:џџџџџџџџџ*
T0
Б
gradients/Mean_grad/ReshapeReshapegradients/mul_1_grad/Reshape!gradients/Mean_grad/DynamicStitch*
_class
	loc:@Mean*
_output_shapes
:*
Tshape0*
T0
Љ
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*
_class
	loc:@Mean*

Tmultiples0*
_output_shapes
:*
T0
w
gradients/Mean_grad/Shape_2ShapeNeg*
_class
	loc:@Mean*
_output_shapes
:*
out_type0*
T0
x
gradients/Mean_grad/Shape_3ShapeMean*
_class
	loc:@Mean*
_output_shapes
:*
out_type0*
T0
|
gradients/Mean_grad/ConstConst*
_class
	loc:@Mean*
_output_shapes
:*
dtype0*
valueB: 
Џ
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
_class
	loc:@Mean*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
~
gradients/Mean_grad/Const_1Const*
_class
	loc:@Mean*
_output_shapes
:*
dtype0*
valueB: 
Г
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
_class
	loc:@Mean*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
z
gradients/Mean_grad/Maximum_1/yConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B :

gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
_class
	loc:@Mean*
_output_shapes
: *
T0

gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
_class
	loc:@Mean*
_output_shapes
: *
T0

gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

DstT0*
_class
	loc:@Mean*

SrcT0*
_output_shapes
: 
Ё
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_class
	loc:@Mean*#
_output_shapes
:џџџџџџџџџ*
T0

gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*
_class

loc:@Neg*#
_output_shapes
:џџџџџџџџџ*
T0
w
gradients/Sum_1_grad/ShapeShapemul*
_class

loc:@Sum_1*
_output_shapes
:*
out_type0*
T0
u
gradients/Sum_1_grad/SizeConst*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
value	B :

gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
_class

loc:@Sum_1*
_output_shapes
: *
T0

gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
_class

loc:@Sum_1*
_output_shapes
: *
T0
y
gradients/Sum_1_grad/Shape_1Const*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
valueB 
|
 gradients/Sum_1_grad/range/startConst*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
value	B : 
|
 gradients/Sum_1_grad/range/deltaConst*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
value	B :
Ф
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*
_class

loc:@Sum_1*
_output_shapes
:*

Tidx0
{
gradients/Sum_1_grad/Fill/valueConst*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
value	B :

gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
_class

loc:@Sum_1*
_output_shapes
: *
T0
ё
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
_class

loc:@Sum_1*#
_output_shapes
:џџџџџџџџџ*
N*
T0
z
gradients/Sum_1_grad/Maximum/yConst*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
value	B :
Г
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
_class

loc:@Sum_1*#
_output_shapes
:џџџџџџџџџ*
T0
Ђ
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
_class

loc:@Sum_1*
_output_shapes
:*
T0
Ў
gradients/Sum_1_grad/ReshapeReshapegradients/Neg_grad/Neg"gradients/Sum_1_grad/DynamicStitch*
_class

loc:@Sum_1*
_output_shapes
:*
Tshape0*
T0
М
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*
_class

loc:@Sum_1*

Tmultiples0*'
_output_shapes
:џџџџџџџџџ*
T0
~
gradients/mul_grad/ShapeShapedense_4_target*
_class

loc:@mul*
_output_shapes
:*
out_type0*
T0
u
gradients/mul_grad/Shape_1ShapeLog*
_class

loc:@mul*
_output_shapes
:*
out_type0*
T0
Ь
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
_class

loc:@mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

gradients/mul_grad/mulMulgradients/Sum_1_grad/TileLog*
_class

loc:@mul*'
_output_shapes
:џџџџџџџџџ*
T0
З
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_class

loc:@mul*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
И
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
_class

loc:@mul*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Tshape0*
T0

gradients/mul_grad/mul_1Muldense_4_targetgradients/Sum_1_grad/Tile*
_class

loc:@mul*'
_output_shapes
:џџџџџџџџџ*
T0
Н
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_class

loc:@mul*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Е
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
_class

loc:@mul*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
Ѓ
gradients/Log_grad/Reciprocal
Reciprocalclip_by_value^gradients/mul_grad/Reshape_1*
_class

loc:@Log*'
_output_shapes
:џџџџџџџџџ*
T0
Є
gradients/Log_grad/mulMulgradients/mul_grad/Reshape_1gradients/Log_grad/Reciprocal*
_class

loc:@Log*'
_output_shapes
:џџџџџџџџџ*
T0

"gradients/clip_by_value_grad/ShapeShapeclip_by_value/Minimum* 
_class
loc:@clip_by_value*
_output_shapes
:*
out_type0*
T0

$gradients/clip_by_value_grad/Shape_1Const* 
_class
loc:@clip_by_value*
_output_shapes
: *
dtype0*
valueB 

$gradients/clip_by_value_grad/Shape_2Shapegradients/Log_grad/mul* 
_class
loc:@clip_by_value*
_output_shapes
:*
out_type0*
T0

(gradients/clip_by_value_grad/zeros/ConstConst* 
_class
loc:@clip_by_value*
_output_shapes
: *
dtype0*
valueB
 *    
Ю
"gradients/clip_by_value_grad/zerosFill$gradients/clip_by_value_grad/Shape_2(gradients/clip_by_value_grad/zeros/Const* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
Ћ
)gradients/clip_by_value_grad/GreaterEqualGreaterEqualclip_by_value/MinimumConst* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
є
2gradients/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/clip_by_value_grad/Shape$gradients/clip_by_value_grad/Shape_1* 
_class
loc:@clip_by_value*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ш
#gradients/clip_by_value_grad/SelectSelect)gradients/clip_by_value_grad/GreaterEqualgradients/Log_grad/mul"gradients/clip_by_value_grad/zeros* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
Ћ
'gradients/clip_by_value_grad/LogicalNot
LogicalNot)gradients/clip_by_value_grad/GreaterEqual* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ
ш
%gradients/clip_by_value_grad/Select_1Select'gradients/clip_by_value_grad/LogicalNotgradients/Log_grad/mul"gradients/clip_by_value_grad/zeros* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
т
 gradients/clip_by_value_grad/SumSum#gradients/clip_by_value_grad/Select2gradients/clip_by_value_grad/BroadcastGradientArgs* 
_class
loc:@clip_by_value*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
з
$gradients/clip_by_value_grad/ReshapeReshape gradients/clip_by_value_grad/Sum"gradients/clip_by_value_grad/Shape* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
ш
"gradients/clip_by_value_grad/Sum_1Sum%gradients/clip_by_value_grad/Select_14gradients/clip_by_value_grad/BroadcastGradientArgs:1* 
_class
loc:@clip_by_value*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Ь
&gradients/clip_by_value_grad/Reshape_1Reshape"gradients/clip_by_value_grad/Sum_1$gradients/clip_by_value_grad/Shape_1* 
_class
loc:@clip_by_value*
_output_shapes
: *
Tshape0*
T0

*gradients/clip_by_value/Minimum_grad/ShapeShapetruediv*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
:*
out_type0*
T0

,gradients/clip_by_value/Minimum_grad/Shape_1Const*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
: *
dtype0*
valueB 
К
,gradients/clip_by_value/Minimum_grad/Shape_2Shape$gradients/clip_by_value_grad/Reshape*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
:*
out_type0*
T0

0gradients/clip_by_value/Minimum_grad/zeros/ConstConst*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
: *
dtype0*
valueB
 *    
ю
*gradients/clip_by_value/Minimum_grad/zerosFill,gradients/clip_by_value/Minimum_grad/Shape_20gradients/clip_by_value/Minimum_grad/zeros/Const*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0
Ѕ
.gradients/clip_by_value/Minimum_grad/LessEqual	LessEqualtruedivsub*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0

:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/clip_by_value/Minimum_grad/Shape,gradients/clip_by_value/Minimum_grad/Shape_1*(
_class
loc:@clip_by_value/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

+gradients/clip_by_value/Minimum_grad/SelectSelect.gradients/clip_by_value/Minimum_grad/LessEqual$gradients/clip_by_value_grad/Reshape*gradients/clip_by_value/Minimum_grad/zeros*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0
Р
/gradients/clip_by_value/Minimum_grad/LogicalNot
LogicalNot.gradients/clip_by_value/Minimum_grad/LessEqual*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ

-gradients/clip_by_value/Minimum_grad/Select_1Select/gradients/clip_by_value/Minimum_grad/LogicalNot$gradients/clip_by_value_grad/Reshape*gradients/clip_by_value/Minimum_grad/zeros*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0

(gradients/clip_by_value/Minimum_grad/SumSum+gradients/clip_by_value/Minimum_grad/Select:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs*(
_class
loc:@clip_by_value/Minimum*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
ї
,gradients/clip_by_value/Minimum_grad/ReshapeReshape(gradients/clip_by_value/Minimum_grad/Sum*gradients/clip_by_value/Minimum_grad/Shape*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0

*gradients/clip_by_value/Minimum_grad/Sum_1Sum-gradients/clip_by_value/Minimum_grad/Select_1<gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*(
_class
loc:@clip_by_value/Minimum*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
ь
.gradients/clip_by_value/Minimum_grad/Reshape_1Reshape*gradients/clip_by_value/Minimum_grad/Sum_1,gradients/clip_by_value/Minimum_grad/Shape_1*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
: *
Tshape0*
T0

gradients/truediv_grad/ShapeShapedense_4/Softmax*
_class
loc:@truediv*
_output_shapes
:*
out_type0*
T0
}
gradients/truediv_grad/Shape_1ShapeSum*
_class
loc:@truediv*
_output_shapes
:*
out_type0*
T0
м
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
_class
loc:@truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Њ
gradients/truediv_grad/RealDivRealDiv,gradients/clip_by_value/Minimum_grad/ReshapeSum*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
T0
Ы
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
_class
loc:@truediv*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
П
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0

gradients/truediv_grad/NegNegdense_4/Softmax*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
T0

 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegSum*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
T0
 
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Sum*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
T0
П
gradients/truediv_grad/mulMul,gradients/clip_by_value/Minimum_grad/Reshape gradients/truediv_grad/RealDiv_2*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
T0
Ы
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
_class
loc:@truediv*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Х
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0

gradients/Sum_grad/ShapeShapedense_4/Softmax*
_class

loc:@Sum*
_output_shapes
:*
out_type0*
T0
q
gradients/Sum_grad/SizeConst*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
value	B :

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
_class

loc:@Sum*
_output_shapes
: *
T0

gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
_class

loc:@Sum*
_output_shapes
: *
T0
u
gradients/Sum_grad/Shape_1Const*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
valueB 
x
gradients/Sum_grad/range/startConst*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
value	B : 
x
gradients/Sum_grad/range/deltaConst*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
value	B :
К
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*
_class

loc:@Sum*
_output_shapes
:*

Tidx0
w
gradients/Sum_grad/Fill/valueConst*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
value	B :

gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
_class

loc:@Sum*
_output_shapes
: *
T0
х
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
_class

loc:@Sum*#
_output_shapes
:џџџџџџџџџ*
N*
T0
v
gradients/Sum_grad/Maximum/yConst*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
value	B :
Ћ
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
_class

loc:@Sum*#
_output_shapes
:џџџџџџџџџ*
T0

gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
_class

loc:@Sum*
_output_shapes
:*
T0
В
gradients/Sum_grad/ReshapeReshape gradients/truediv_grad/Reshape_1 gradients/Sum_grad/DynamicStitch*
_class

loc:@Sum*
_output_shapes
:*
Tshape0*
T0
Д
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
_class

loc:@Sum*

Tmultiples0*'
_output_shapes
:џџџџџџџџџ*
T0
І
gradients/AddNAddNgradients/truediv_grad/Reshapegradients/Sum_grad/Tile*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
N*
T0
 
"gradients/dense_4/Softmax_grad/mulMulgradients/AddNdense_4/Softmax*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0
Ђ
4gradients/dense_4/Softmax_grad/Sum/reduction_indicesConst*"
_class
loc:@dense_4/Softmax*
_output_shapes
:*
dtype0*
valueB:
ђ
"gradients/dense_4/Softmax_grad/SumSum"gradients/dense_4/Softmax_grad/mul4gradients/dense_4/Softmax_grad/Sum/reduction_indices*"
_class
loc:@dense_4/Softmax*
	keep_dims( *#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
Ё
,gradients/dense_4/Softmax_grad/Reshape/shapeConst*"
_class
loc:@dense_4/Softmax*
_output_shapes
:*
dtype0*
valueB"џџџџ   
ч
&gradients/dense_4/Softmax_grad/ReshapeReshape"gradients/dense_4/Softmax_grad/Sum,gradients/dense_4/Softmax_grad/Reshape/shape*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
З
"gradients/dense_4/Softmax_grad/subSubgradients/AddN&gradients/dense_4/Softmax_grad/Reshape*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0
Ж
$gradients/dense_4/Softmax_grad/mul_1Mul"gradients/dense_4/Softmax_grad/subdense_4/Softmax*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0
П
*gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_4/Softmax_grad/mul_1*"
_class
loc:@dense_4/BiasAdd*
_output_shapes
:*
data_formatNHWC*
T0
ф
$gradients/dense_4/MatMul_grad/MatMulMatMul$gradients/dense_4/Softmax_grad/mul_1dense_4/kernel/read*!
_class
loc:@dense_4/MatMul*'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
transpose_a( *
T0
ж
&gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Tanh$gradients/dense_4/Softmax_grad/mul_1*!
_class
loc:@dense_4/MatMul*
_output_shapes

:*
transpose_b( *
transpose_a(*
T0
З
$gradients/dense_3/Tanh_grad/TanhGradTanhGraddense_3/Tanh$gradients/dense_4/MatMul_grad/MatMul*
_class
loc:@dense_3/Tanh*'
_output_shapes
:џџџџџџџџџ*
T0
П
*gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_3/Tanh_grad/TanhGrad*"
_class
loc:@dense_3/BiasAdd*
_output_shapes
:*
data_formatNHWC*
T0
ф
$gradients/dense_3/MatMul_grad/MatMulMatMul$gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*!
_class
loc:@dense_3/MatMul*'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
transpose_a( *
T0
ж
&gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Tanh$gradients/dense_3/Tanh_grad/TanhGrad*!
_class
loc:@dense_3/MatMul*
_output_shapes

:*
transpose_b( *
transpose_a(*
T0
З
$gradients/dense_2/Tanh_grad/TanhGradTanhGraddense_2/Tanh$gradients/dense_3/MatMul_grad/MatMul*
_class
loc:@dense_2/Tanh*'
_output_shapes
:џџџџџџџџџ*
T0
П
*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_2/Tanh_grad/TanhGrad*"
_class
loc:@dense_2/BiasAdd*
_output_shapes
:*
data_formatNHWC*
T0
ф
$gradients/dense_2/MatMul_grad/MatMulMatMul$gradients/dense_2/Tanh_grad/TanhGraddense_2/kernel/read*!
_class
loc:@dense_2/MatMul*'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
transpose_a( *
T0
ж
&gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh$gradients/dense_2/Tanh_grad/TanhGrad*!
_class
loc:@dense_2/MatMul*
_output_shapes

:*
transpose_b( *
transpose_a(*
T0
З
$gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanh$gradients/dense_2/MatMul_grad/MatMul*
_class
loc:@dense_1/Tanh*'
_output_shapes
:џџџџџџџџџ*
T0
П
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_1/Tanh_grad/TanhGrad*"
_class
loc:@dense_1/BiasAdd*
_output_shapes
:*
data_formatNHWC*
T0
ф
$gradients/dense_1/MatMul_grad/MatMulMatMul$gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*!
_class
loc:@dense_1/MatMul*'
_output_shapes
:џџџџџџџџџ*
transpose_b(*
transpose_a( *
T0
б
&gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_1$gradients/dense_1/Tanh_grad/TanhGrad*!
_class
loc:@dense_1/MatMul*
_output_shapes

:*
transpose_b( *
transpose_a(*
T0
T
Const_4Const*
_output_shapes
:*
dtype0*
valueB*    
t
Variable
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 

Variable/AssignAssignVariableConst_4*
use_locking(*
validate_shape(*
_class
loc:@Variable*
_output_shapes
:*
T0
e
Variable/readIdentityVariable*
_class
loc:@Variable*
_output_shapes
:*
T0
\
Const_5Const*
_output_shapes

:*
dtype0*
valueB*    
~

Variable_1
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
Ё
Variable_1/AssignAssign
Variable_1Const_5*
use_locking(*
validate_shape(*
_class
loc:@Variable_1*
_output_shapes

:*
T0
o
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
_output_shapes

:*
T0
T
Const_6Const*
_output_shapes
:*
dtype0*
valueB*    
v

Variable_2
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 

Variable_2/AssignAssign
Variable_2Const_6*
use_locking(*
validate_shape(*
_class
loc:@Variable_2*
_output_shapes
:*
T0
k
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
_output_shapes
:*
T0
\
Const_7Const*
_output_shapes

:*
dtype0*
valueB*    
~

Variable_3
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
Ё
Variable_3/AssignAssign
Variable_3Const_7*
use_locking(*
validate_shape(*
_class
loc:@Variable_3*
_output_shapes

:*
T0
o
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
_output_shapes

:*
T0
T
Const_8Const*
_output_shapes
:*
dtype0*
valueB*    
v

Variable_4
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 

Variable_4/AssignAssign
Variable_4Const_8*
use_locking(*
validate_shape(*
_class
loc:@Variable_4*
_output_shapes
:*
T0
k
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
_output_shapes
:*
T0
\
Const_9Const*
_output_shapes

:*
dtype0*
valueB*    
~

Variable_5
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
Ё
Variable_5/AssignAssign
Variable_5Const_9*
use_locking(*
validate_shape(*
_class
loc:@Variable_5*
_output_shapes

:*
T0
o
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
_output_shapes

:*
T0
U
Const_10Const*
_output_shapes
:*
dtype0*
valueB*    
v

Variable_6
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 

Variable_6/AssignAssign
Variable_6Const_10*
use_locking(*
validate_shape(*
_class
loc:@Variable_6*
_output_shapes
:*
T0
k
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
_output_shapes
:*
T0
]
Const_11Const*
_output_shapes

:*
dtype0*
valueB*    
~

Variable_7
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
Ђ
Variable_7/AssignAssign
Variable_7Const_11*
use_locking(*
validate_shape(*
_class
loc:@Variable_7*
_output_shapes

:*
T0
o
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
_output_shapes

:*
T0
J
mul_3Mulrho/readVariable/read*
_output_shapes
:*
T0
L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@
sub_1Subsub_1/xrho/read*
_output_shapes
: *
T0
a
SquareSquare*gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
@
mul_4Mulsub_1Square*
_output_shapes
:*
T0
=
addAddmul_3mul_4*
_output_shapes
:*
T0

AssignAssignVariableadd*
use_locking(*
validate_shape(*
_class
loc:@Variable*
_output_shapes
:*
T0
f
mul_5Mullr/read*gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
M
Const_12Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_13Const*
_output_shapes
: *
dtype0*
valueB
 *  
V
clip_by_value_1/MinimumMinimumaddConst_13*
_output_shapes
:*
T0
b
clip_by_value_1Maximumclip_by_value_1/MinimumConst_12*
_output_shapes
:*
T0
B
SqrtSqrtclip_by_value_1*
_output_shapes
:*
T0
L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
@
add_1AddSqrtadd_1/y*
_output_shapes
:*
T0
G
	truediv_2RealDivmul_5add_1*
_output_shapes
:*
T0
O
sub_2Subdense_1/bias/read	truediv_2*
_output_shapes
:*
T0

Assign_1Assigndense_1/biassub_2*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias*
_output_shapes
:*
T0
P
mul_6Mulrho/readVariable_1/read*
_output_shapes

:*
T0
L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@
sub_3Subsub_3/xrho/read*
_output_shapes
: *
T0
c
Square_1Square&gradients/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
F
mul_7Mulsub_3Square_1*
_output_shapes

:*
T0
C
add_2Addmul_6mul_7*
_output_shapes

:*
T0

Assign_2Assign
Variable_1add_2*
use_locking(*
validate_shape(*
_class
loc:@Variable_1*
_output_shapes

:*
T0
f
mul_8Mullr/read&gradients/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
M
Const_14Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_15Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_2/MinimumMinimumadd_2Const_15*
_output_shapes

:*
T0
f
clip_by_value_2Maximumclip_by_value_2/MinimumConst_14*
_output_shapes

:*
T0
H
Sqrt_1Sqrtclip_by_value_2*
_output_shapes

:*
T0
L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
F
add_3AddSqrt_1add_3/y*
_output_shapes

:*
T0
K
	truediv_3RealDivmul_8add_3*
_output_shapes

:*
T0
U
sub_4Subdense_1/kernel/read	truediv_3*
_output_shapes

:*
T0

Assign_3Assigndense_1/kernelsub_4*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel*
_output_shapes

:*
T0
L
mul_9Mulrho/readVariable_2/read*
_output_shapes
:*
T0
L
sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@
sub_5Subsub_5/xrho/read*
_output_shapes
: *
T0
c
Square_2Square*gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
C
mul_10Mulsub_5Square_2*
_output_shapes
:*
T0
@
add_4Addmul_9mul_10*
_output_shapes
:*
T0

Assign_4Assign
Variable_2add_4*
use_locking(*
validate_shape(*
_class
loc:@Variable_2*
_output_shapes
:*
T0
g
mul_11Mullr/read*gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
M
Const_16Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_17Const*
_output_shapes
: *
dtype0*
valueB
 *  
X
clip_by_value_3/MinimumMinimumadd_4Const_17*
_output_shapes
:*
T0
b
clip_by_value_3Maximumclip_by_value_3/MinimumConst_16*
_output_shapes
:*
T0
D
Sqrt_2Sqrtclip_by_value_3*
_output_shapes
:*
T0
L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
B
add_5AddSqrt_2add_5/y*
_output_shapes
:*
T0
H
	truediv_4RealDivmul_11add_5*
_output_shapes
:*
T0
O
sub_6Subdense_2/bias/read	truediv_4*
_output_shapes
:*
T0

Assign_5Assigndense_2/biassub_6*
use_locking(*
validate_shape(*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0
Q
mul_12Mulrho/readVariable_3/read*
_output_shapes

:*
T0
L
sub_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@
sub_7Subsub_7/xrho/read*
_output_shapes
: *
T0
c
Square_3Square&gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
G
mul_13Mulsub_7Square_3*
_output_shapes

:*
T0
E
add_6Addmul_12mul_13*
_output_shapes

:*
T0

Assign_6Assign
Variable_3add_6*
use_locking(*
validate_shape(*
_class
loc:@Variable_3*
_output_shapes

:*
T0
g
mul_14Mullr/read&gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
M
Const_18Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_19Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_4/MinimumMinimumadd_6Const_19*
_output_shapes

:*
T0
f
clip_by_value_4Maximumclip_by_value_4/MinimumConst_18*
_output_shapes

:*
T0
H
Sqrt_3Sqrtclip_by_value_4*
_output_shapes

:*
T0
L
add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
F
add_7AddSqrt_3add_7/y*
_output_shapes

:*
T0
L
	truediv_5RealDivmul_14add_7*
_output_shapes

:*
T0
U
sub_8Subdense_2/kernel/read	truediv_5*
_output_shapes

:*
T0

Assign_7Assigndense_2/kernelsub_8*
use_locking(*
validate_shape(*!
_class
loc:@dense_2/kernel*
_output_shapes

:*
T0
M
mul_15Mulrho/readVariable_4/read*
_output_shapes
:*
T0
L
sub_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@
sub_9Subsub_9/xrho/read*
_output_shapes
: *
T0
c
Square_4Square*gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
C
mul_16Mulsub_9Square_4*
_output_shapes
:*
T0
A
add_8Addmul_15mul_16*
_output_shapes
:*
T0

Assign_8Assign
Variable_4add_8*
use_locking(*
validate_shape(*
_class
loc:@Variable_4*
_output_shapes
:*
T0
g
mul_17Mullr/read*gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
M
Const_20Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_21Const*
_output_shapes
: *
dtype0*
valueB
 *  
X
clip_by_value_5/MinimumMinimumadd_8Const_21*
_output_shapes
:*
T0
b
clip_by_value_5Maximumclip_by_value_5/MinimumConst_20*
_output_shapes
:*
T0
D
Sqrt_4Sqrtclip_by_value_5*
_output_shapes
:*
T0
L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
B
add_9AddSqrt_4add_9/y*
_output_shapes
:*
T0
H
	truediv_6RealDivmul_17add_9*
_output_shapes
:*
T0
P
sub_10Subdense_3/bias/read	truediv_6*
_output_shapes
:*
T0

Assign_9Assigndense_3/biassub_10*
use_locking(*
validate_shape(*
_class
loc:@dense_3/bias*
_output_shapes
:*
T0
Q
mul_18Mulrho/readVariable_5/read*
_output_shapes

:*
T0
M
sub_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
B
sub_11Subsub_11/xrho/read*
_output_shapes
: *
T0
c
Square_5Square&gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
H
mul_19Mulsub_11Square_5*
_output_shapes

:*
T0
F
add_10Addmul_18mul_19*
_output_shapes

:*
T0

	Assign_10Assign
Variable_5add_10*
use_locking(*
validate_shape(*
_class
loc:@Variable_5*
_output_shapes

:*
T0
g
mul_20Mullr/read&gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
M
Const_22Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_23Const*
_output_shapes
: *
dtype0*
valueB
 *  
]
clip_by_value_6/MinimumMinimumadd_10Const_23*
_output_shapes

:*
T0
f
clip_by_value_6Maximumclip_by_value_6/MinimumConst_22*
_output_shapes

:*
T0
H
Sqrt_5Sqrtclip_by_value_6*
_output_shapes

:*
T0
M
add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
H
add_11AddSqrt_5add_11/y*
_output_shapes

:*
T0
M
	truediv_7RealDivmul_20add_11*
_output_shapes

:*
T0
V
sub_12Subdense_3/kernel/read	truediv_7*
_output_shapes

:*
T0
 
	Assign_11Assigndense_3/kernelsub_12*
use_locking(*
validate_shape(*!
_class
loc:@dense_3/kernel*
_output_shapes

:*
T0
M
mul_21Mulrho/readVariable_6/read*
_output_shapes
:*
T0
M
sub_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
B
sub_13Subsub_13/xrho/read*
_output_shapes
: *
T0
c
Square_6Square*gradients/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
D
mul_22Mulsub_13Square_6*
_output_shapes
:*
T0
B
add_12Addmul_21mul_22*
_output_shapes
:*
T0

	Assign_12Assign
Variable_6add_12*
use_locking(*
validate_shape(*
_class
loc:@Variable_6*
_output_shapes
:*
T0
g
mul_23Mullr/read*gradients/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
M
Const_24Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_25Const*
_output_shapes
: *
dtype0*
valueB
 *  
Y
clip_by_value_7/MinimumMinimumadd_12Const_25*
_output_shapes
:*
T0
b
clip_by_value_7Maximumclip_by_value_7/MinimumConst_24*
_output_shapes
:*
T0
D
Sqrt_6Sqrtclip_by_value_7*
_output_shapes
:*
T0
M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
D
add_13AddSqrt_6add_13/y*
_output_shapes
:*
T0
I
	truediv_8RealDivmul_23add_13*
_output_shapes
:*
T0
P
sub_14Subdense_4/bias/read	truediv_8*
_output_shapes
:*
T0

	Assign_13Assigndense_4/biassub_14*
use_locking(*
validate_shape(*
_class
loc:@dense_4/bias*
_output_shapes
:*
T0
Q
mul_24Mulrho/readVariable_7/read*
_output_shapes

:*
T0
M
sub_15/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
B
sub_15Subsub_15/xrho/read*
_output_shapes
: *
T0
c
Square_7Square&gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
H
mul_25Mulsub_15Square_7*
_output_shapes

:*
T0
F
add_14Addmul_24mul_25*
_output_shapes

:*
T0

	Assign_14Assign
Variable_7add_14*
use_locking(*
validate_shape(*
_class
loc:@Variable_7*
_output_shapes

:*
T0
g
mul_26Mullr/read&gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
M
Const_26Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_27Const*
_output_shapes
: *
dtype0*
valueB
 *  
]
clip_by_value_8/MinimumMinimumadd_14Const_27*
_output_shapes

:*
T0
f
clip_by_value_8Maximumclip_by_value_8/MinimumConst_26*
_output_shapes

:*
T0
H
Sqrt_7Sqrtclip_by_value_8*
_output_shapes

:*
T0
M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
H
add_15AddSqrt_7add_15/y*
_output_shapes

:*
T0
M
	truediv_9RealDivmul_26add_15*
_output_shapes

:*
T0
V
sub_16Subdense_4/kernel/read	truediv_9*
_output_shapes

:*
T0
 
	Assign_15Assigndense_4/kernelsub_16*
use_locking(*
validate_shape(*!
_class
loc:@dense_4/kernel*
_output_shapes

:*
T0
й
group_deps_1NoOp^mul_2^Mean_3^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15

initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign
^lr/Assign^rho/Assign^decay/Assign^iterations/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign"ЗM~Џ
     yуiС	mМ=жAJЂ
р
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
A
Equal
x"T
y"T
z
"
Ttype:
2	

4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
<
	LessEqual
x"T
y"T
z
"
Ttype:
2		
+
Log
x"T
y"T"
Ttype:	
2


LogicalNot
x

y

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
:
Minimum
x"T
y"T
z"T"
Ttype:	
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
D
NotEqual
x"T
y"T
z
"
Ttype:
2	

A
Placeholder
output"dtype"
dtypetype"
shapeshape: 

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
4

Reciprocal
x"T
y"T"
Ttype:
	2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
8
Softmax
logits"T
softmax"T"
Ttype:
2
,
Sqrt
x"T
y"T"
Ttype:	
2
0
Square
x"T
y"T"
Ttype:
	2	
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
,
Tanh
x"T
y"T"
Ttype:	
2
8
TanhGrad
x"T
y"T
z"T"
Ttype:	
2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring *1.1.02v1.1.0-rc0-61-g1ec6ed5ыч
W
keras_learning_phasePlaceholder*
_output_shapes
:*
shape: *
dtype0

Y
input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
shape: *
dtype0
o
dense_1/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
b
dense_1/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
d
dense_1/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ЋвШ>
А
(dense_1/truncated_normal/TruncatedNormalTruncatedNormaldense_1/truncated_normal/shape*
seed2юН*
_output_shapes

:*
seedБџх)*
dtype0*
T0

dense_1/truncated_normal/mulMul(dense_1/truncated_normal/TruncatedNormaldense_1/truncated_normal/stddev*
_output_shapes

:*
T0

dense_1/truncated_normalAdddense_1/truncated_normal/muldense_1/truncated_normal/mean*
_output_shapes

:*
T0

dense_1/kernel
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
О
dense_1/kernel/AssignAssigndense_1/kerneldense_1/truncated_normal*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel*
_output_shapes

:*
T0
{
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
_output_shapes

:*
T0
Z
dense_1/ConstConst*
_output_shapes
:*
dtype0*
valueB*    
x
dense_1/bias
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
Љ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias*
_output_shapes
:*
T0
q
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
_output_shapes
:*
T0

dense_1/MatMulMatMulinput_1dense_1/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC*
T0
W
dense_1/TanhTanhdense_1/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
o
dense_2/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
b
dense_2/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
d
dense_2/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
А
(dense_2/truncated_normal/TruncatedNormalTruncatedNormaldense_2/truncated_normal/shape*
seed2ў*
_output_shapes

:*
seedБџх)*
dtype0*
T0

dense_2/truncated_normal/mulMul(dense_2/truncated_normal/TruncatedNormaldense_2/truncated_normal/stddev*
_output_shapes

:*
T0

dense_2/truncated_normalAdddense_2/truncated_normal/muldense_2/truncated_normal/mean*
_output_shapes

:*
T0

dense_2/kernel
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
О
dense_2/kernel/AssignAssigndense_2/kerneldense_2/truncated_normal*
use_locking(*
validate_shape(*!
_class
loc:@dense_2/kernel*
_output_shapes

:*
T0
{
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
_output_shapes

:*
T0
Z
dense_2/ConstConst*
_output_shapes
:*
dtype0*
valueB*    
x
dense_2/bias
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
Љ
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
use_locking(*
validate_shape(*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0
q
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0

dense_2/MatMulMatMuldense_1/Tanhdense_2/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC*
T0
W
dense_2/TanhTanhdense_2/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
o
dense_3/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
b
dense_3/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
d
dense_3/truncated_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *ш!?
Џ
(dense_3/truncated_normal/TruncatedNormalTruncatedNormaldense_3/truncated_normal/shape*
seed2щX*
_output_shapes

:*
seedБџх)*
dtype0*
T0

dense_3/truncated_normal/mulMul(dense_3/truncated_normal/TruncatedNormaldense_3/truncated_normal/stddev*
_output_shapes

:*
T0

dense_3/truncated_normalAdddense_3/truncated_normal/muldense_3/truncated_normal/mean*
_output_shapes

:*
T0

dense_3/kernel
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
О
dense_3/kernel/AssignAssigndense_3/kerneldense_3/truncated_normal*
use_locking(*
validate_shape(*!
_class
loc:@dense_3/kernel*
_output_shapes

:*
T0
{
dense_3/kernel/readIdentitydense_3/kernel*!
_class
loc:@dense_3/kernel*
_output_shapes

:*
T0
Z
dense_3/ConstConst*
_output_shapes
:*
dtype0*
valueB*    
x
dense_3/bias
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
Љ
dense_3/bias/AssignAssigndense_3/biasdense_3/Const*
use_locking(*
validate_shape(*
_class
loc:@dense_3/bias*
_output_shapes
:*
T0
q
dense_3/bias/readIdentitydense_3/bias*
_class
loc:@dense_3/bias*
_output_shapes
:*
T0

dense_3/MatMulMatMuldense_2/Tanhdense_3/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_3/BiasAddBiasAdddense_3/MatMuldense_3/bias/read*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC*
T0
W
dense_3/TanhTanhdense_3/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
m
dense_4/random_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
_
dense_4/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *7П
_
dense_4/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *7?
Ї
$dense_4/random_uniform/RandomUniformRandomUniformdense_4/random_uniform/shape*
seed2їЛ9*
_output_shapes

:*
seedБџх)*
dtype0*
T0
z
dense_4/random_uniform/subSubdense_4/random_uniform/maxdense_4/random_uniform/min*
_output_shapes
: *
T0

dense_4/random_uniform/mulMul$dense_4/random_uniform/RandomUniformdense_4/random_uniform/sub*
_output_shapes

:*
T0
~
dense_4/random_uniformAdddense_4/random_uniform/muldense_4/random_uniform/min*
_output_shapes

:*
T0

dense_4/kernel
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
М
dense_4/kernel/AssignAssigndense_4/kerneldense_4/random_uniform*
use_locking(*
validate_shape(*!
_class
loc:@dense_4/kernel*
_output_shapes

:*
T0
{
dense_4/kernel/readIdentitydense_4/kernel*!
_class
loc:@dense_4/kernel*
_output_shapes

:*
T0
Z
dense_4/ConstConst*
_output_shapes
:*
dtype0*
valueB*    
x
dense_4/bias
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 
Љ
dense_4/bias/AssignAssigndense_4/biasdense_4/Const*
use_locking(*
validate_shape(*
_class
loc:@dense_4/bias*
_output_shapes
:*
T0
q
dense_4/bias/readIdentitydense_4/bias*
_class
loc:@dense_4/bias*
_output_shapes
:*
T0

dense_4/MatMulMatMuldense_3/Tanhdense_4/kernel/read*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b( *
T0

dense_4/BiasAddBiasAdddense_4/MatMuldense_4/bias/read*'
_output_shapes
:џџџџџџџџџ*
data_formatNHWC*
T0
]
dense_4/SoftmaxSoftmaxdense_4/BiasAdd*'
_output_shapes
:џџџџџџџџџ*
T0
U
lr/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL=
f
lr
VariableV2*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 

	lr/AssignAssignlrlr/initial_value*
use_locking(*
validate_shape(*
_class
	loc:@lr*
_output_shapes
: *
T0
O
lr/readIdentitylr*
_class
	loc:@lr*
_output_shapes
: *
T0
V
rho/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *fff?
g
rho
VariableV2*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 


rho/AssignAssignrhorho/initial_value*
use_locking(*
validate_shape(*
_class

loc:@rho*
_output_shapes
: *
T0
R
rho/readIdentityrho*
_class

loc:@rho*
_output_shapes
: *
T0
X
decay/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
i
decay
VariableV2*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 

decay/AssignAssigndecaydecay/initial_value*
use_locking(*
validate_shape(*
_class

loc:@decay*
_output_shapes
: *
T0
X

decay/readIdentitydecay*
_class

loc:@decay*
_output_shapes
: *
T0
]
iterations/initial_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
n

iterations
VariableV2*
_output_shapes
: *
shape: *
	container *
dtype0*
shared_name 
Њ
iterations/AssignAssign
iterationsiterations/initial_value*
use_locking(*
validate_shape(*
_class
loc:@iterations*
_output_shapes
: *
T0
g
iterations/readIdentity
iterations*
_class
loc:@iterations*
_output_shapes
: *
T0
d
dense_4_sample_weightsPlaceholder*#
_output_shapes
:џџџџџџџџџ*
shape: *
dtype0
i
dense_4_targetPlaceholder*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
shape: *
dtype0
W
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :

SumSumdense_4/SoftmaxSum/reduction_indices*
	keep_dims(*'
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
Z
truedivRealDivdense_4/SoftmaxSum*'
_output_shapes
:џџџџџџџџџ*
T0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
9
subSubsub/xConst*
_output_shapes
: *
T0
`
clip_by_value/MinimumMinimumtruedivsub*'
_output_shapes
:џџџџџџџџџ*
T0
h
clip_by_valueMaximumclip_by_value/MinimumConst*'
_output_shapes
:џџџџџџџџџ*
T0
K
LogLogclip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
Q
mulMuldense_4_targetLog*'
_output_shapes
:џџџџџџџџџ*
T0
Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
u
Sum_1SummulSum_1/reduction_indices*
	keep_dims( *#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
?
NegNegSum_1*#
_output_shapes
:џџџџџџџџџ*
T0
Y
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB 
t
MeanMeanNegMean/reduction_indices*
	keep_dims( *#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
X
mul_1MulMeandense_4_sample_weights*#
_output_shapes
:џџџџџџџџџ*
T0
O

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
f
NotEqualNotEqualdense_4_sample_weights
NotEqual/y*#
_output_shapes
:џџџџџџџџџ*
T0
S
CastCastNotEqual*

DstT0*

SrcT0
*#
_output_shapes
:џџџџџџџџџ
Q
Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
[
Mean_1MeanCastConst_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
Q
	truediv_1RealDivmul_1Mean_1*#
_output_shapes
:џџџџџџџџџ*
T0
Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
`
Mean_2Mean	truediv_1Const_2*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
>
mul_2Mulmul_2/xMean_2*
_output_shapes
: *
T0
R
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
l
ArgMaxArgMaxdense_4_targetArgMax/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
T
ArgMax_1/dimensionConst*
_output_shapes
: *
dtype0*
value	B :
q
ArgMax_1ArgMaxdense_4/SoftmaxArgMax_1/dimension*#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
N
EqualEqualArgMaxArgMax_1*#
_output_shapes
:џџџџџџџџџ*
T0	
R
Cast_1CastEqual*

DstT0*

SrcT0
*#
_output_shapes
:џџџџџџџџџ
Q
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
]
Mean_3MeanCast_1Const_3*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
#

group_depsNoOp^mul_2^Mean_3
l
gradients/ShapeConst*
_class

loc:@mul_2*
_output_shapes
: *
dtype0*
valueB 
n
gradients/ConstConst*
_class

loc:@mul_2*
_output_shapes
: *
dtype0*
valueB
 *  ?
s
gradients/FillFillgradients/Shapegradients/Const*
_class

loc:@mul_2*
_output_shapes
: *
T0
w
gradients/mul_2_grad/ShapeConst*
_class

loc:@mul_2*
_output_shapes
: *
dtype0*
valueB 
y
gradients/mul_2_grad/Shape_1Const*
_class

loc:@mul_2*
_output_shapes
: *
dtype0*
valueB 
д
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
_class

loc:@mul_2*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
r
gradients/mul_2_grad/mulMulgradients/FillMean_2*
_class

loc:@mul_2*
_output_shapes
: *
T0
П
gradients/mul_2_grad/SumSumgradients/mul_2_grad/mul*gradients/mul_2_grad/BroadcastGradientArgs*
_class

loc:@mul_2*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
І
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
_class

loc:@mul_2*
_output_shapes
: *
Tshape0*
T0
u
gradients/mul_2_grad/mul_1Mulmul_2/xgradients/Fill*
_class

loc:@mul_2*
_output_shapes
: *
T0
Х
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
_class

loc:@mul_2*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Ќ
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
_class

loc:@mul_2*
_output_shapes
: *
Tshape0*
T0

#gradients/Mean_2_grad/Reshape/shapeConst*
_class
loc:@Mean_2*
_output_shapes
:*
dtype0*
valueB:
Л
gradients/Mean_2_grad/ReshapeReshapegradients/mul_2_grad/Reshape_1#gradients/Mean_2_grad/Reshape/shape*
_class
loc:@Mean_2*
_output_shapes
:*
Tshape0*
T0

gradients/Mean_2_grad/ShapeShape	truediv_1*
_class
loc:@Mean_2*
_output_shapes
:*
out_type0*
T0
Й
gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Shape*
_class
loc:@Mean_2*

Tmultiples0*#
_output_shapes
:џџџџџџџџџ*
T0

gradients/Mean_2_grad/Shape_1Shape	truediv_1*
_class
loc:@Mean_2*
_output_shapes
:*
out_type0*
T0
{
gradients/Mean_2_grad/Shape_2Const*
_class
loc:@Mean_2*
_output_shapes
: *
dtype0*
valueB 

gradients/Mean_2_grad/ConstConst*
_class
loc:@Mean_2*
_output_shapes
:*
dtype0*
valueB: 
З
gradients/Mean_2_grad/ProdProdgradients/Mean_2_grad/Shape_1gradients/Mean_2_grad/Const*
_class
loc:@Mean_2*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0

gradients/Mean_2_grad/Const_1Const*
_class
loc:@Mean_2*
_output_shapes
:*
dtype0*
valueB: 
Л
gradients/Mean_2_grad/Prod_1Prodgradients/Mean_2_grad/Shape_2gradients/Mean_2_grad/Const_1*
_class
loc:@Mean_2*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
|
gradients/Mean_2_grad/Maximum/yConst*
_class
loc:@Mean_2*
_output_shapes
: *
dtype0*
value	B :
Ѓ
gradients/Mean_2_grad/MaximumMaximumgradients/Mean_2_grad/Prod_1gradients/Mean_2_grad/Maximum/y*
_class
loc:@Mean_2*
_output_shapes
: *
T0
Ё
gradients/Mean_2_grad/floordivFloorDivgradients/Mean_2_grad/Prodgradients/Mean_2_grad/Maximum*
_class
loc:@Mean_2*
_output_shapes
: *
T0

gradients/Mean_2_grad/CastCastgradients/Mean_2_grad/floordiv*

DstT0*
_class
loc:@Mean_2*

SrcT0*
_output_shapes
: 
Љ
gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Cast*
_class
loc:@Mean_2*#
_output_shapes
:џџџџџџџџџ*
T0

gradients/truediv_1_grad/ShapeShapemul_1*
_class
loc:@truediv_1*
_output_shapes
:*
out_type0*
T0

 gradients/truediv_1_grad/Shape_1Const*
_class
loc:@truediv_1*
_output_shapes
: *
dtype0*
valueB 
ф
.gradients/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_1_grad/Shape gradients/truediv_1_grad/Shape_1*
_class
loc:@truediv_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

 gradients/truediv_1_grad/RealDivRealDivgradients/Mean_2_grad/truedivMean_1*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0
г
gradients/truediv_1_grad/SumSum gradients/truediv_1_grad/RealDiv.gradients/truediv_1_grad/BroadcastGradientArgs*
_class
loc:@truediv_1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
У
 gradients/truediv_1_grad/ReshapeReshapegradients/truediv_1_grad/Sumgradients/truediv_1_grad/Shape*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
v
gradients/truediv_1_grad/NegNegmul_1*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0

"gradients/truediv_1_grad/RealDiv_1RealDivgradients/truediv_1_grad/NegMean_1*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0
Ѕ
"gradients/truediv_1_grad/RealDiv_2RealDiv"gradients/truediv_1_grad/RealDiv_1Mean_1*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0
В
gradients/truediv_1_grad/mulMulgradients/Mean_2_grad/truediv"gradients/truediv_1_grad/RealDiv_2*
_class
loc:@truediv_1*#
_output_shapes
:џџџџџџџџџ*
T0
г
gradients/truediv_1_grad/Sum_1Sumgradients/truediv_1_grad/mul0gradients/truediv_1_grad/BroadcastGradientArgs:1*
_class
loc:@truediv_1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
М
"gradients/truediv_1_grad/Reshape_1Reshapegradients/truediv_1_grad/Sum_1 gradients/truediv_1_grad/Shape_1*
_class
loc:@truediv_1*
_output_shapes
: *
Tshape0*
T0
x
gradients/mul_1_grad/ShapeShapeMean*
_class

loc:@mul_1*
_output_shapes
:*
out_type0*
T0

gradients/mul_1_grad/Shape_1Shapedense_4_sample_weights*
_class

loc:@mul_1*
_output_shapes
:*
out_type0*
T0
д
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
_class

loc:@mul_1*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Ё
gradients/mul_1_grad/mulMul gradients/truediv_1_grad/Reshapedense_4_sample_weights*
_class

loc:@mul_1*#
_output_shapes
:џџџџџџџџџ*
T0
П
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
_class

loc:@mul_1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Г
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
_class

loc:@mul_1*#
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0

gradients/mul_1_grad/mul_1MulMean gradients/truediv_1_grad/Reshape*
_class

loc:@mul_1*#
_output_shapes
:џџџџџџџџџ*
T0
Х
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
_class

loc:@mul_1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Й
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
_class

loc:@mul_1*#
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
u
gradients/Mean_grad/ShapeShapeNeg*
_class
	loc:@Mean*
_output_shapes
:*
out_type0*
T0
s
gradients/Mean_grad/SizeConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B :

gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
_class
	loc:@Mean*
_output_shapes
: *
T0

gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
_class
	loc:@Mean*
_output_shapes
: *
T0
~
gradients/Mean_grad/Shape_1Const*
_class
	loc:@Mean*
_output_shapes
:*
dtype0*
valueB: 
z
gradients/Mean_grad/range/startConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B : 
z
gradients/Mean_grad/range/deltaConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B :
П
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*
_class
	loc:@Mean*
_output_shapes
:*

Tidx0
y
gradients/Mean_grad/Fill/valueConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B :

gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
_class
	loc:@Mean*
_output_shapes
: *
T0
ы
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
_class
	loc:@Mean*#
_output_shapes
:џџџџџџџџџ*
N*
T0
x
gradients/Mean_grad/Maximum/yConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B :
Џ
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
_class
	loc:@Mean*#
_output_shapes
:џџџџџџџџџ*
T0
Ї
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
_class
	loc:@Mean*#
_output_shapes
:џџџџџџџџџ*
T0
Б
gradients/Mean_grad/ReshapeReshapegradients/mul_1_grad/Reshape!gradients/Mean_grad/DynamicStitch*
_class
	loc:@Mean*
_output_shapes
:*
Tshape0*
T0
Љ
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*
_class
	loc:@Mean*

Tmultiples0*
_output_shapes
:*
T0
w
gradients/Mean_grad/Shape_2ShapeNeg*
_class
	loc:@Mean*
_output_shapes
:*
out_type0*
T0
x
gradients/Mean_grad/Shape_3ShapeMean*
_class
	loc:@Mean*
_output_shapes
:*
out_type0*
T0
|
gradients/Mean_grad/ConstConst*
_class
	loc:@Mean*
_output_shapes
:*
dtype0*
valueB: 
Џ
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
_class
	loc:@Mean*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
~
gradients/Mean_grad/Const_1Const*
_class
	loc:@Mean*
_output_shapes
:*
dtype0*
valueB: 
Г
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
_class
	loc:@Mean*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
z
gradients/Mean_grad/Maximum_1/yConst*
_class
	loc:@Mean*
_output_shapes
: *
dtype0*
value	B :

gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
_class
	loc:@Mean*
_output_shapes
: *
T0

gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
_class
	loc:@Mean*
_output_shapes
: *
T0

gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

DstT0*
_class
	loc:@Mean*

SrcT0*
_output_shapes
: 
Ё
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_class
	loc:@Mean*#
_output_shapes
:џџџџџџџџџ*
T0

gradients/Neg_grad/NegNeggradients/Mean_grad/truediv*
_class

loc:@Neg*#
_output_shapes
:џџџџџџџџџ*
T0
w
gradients/Sum_1_grad/ShapeShapemul*
_class

loc:@Sum_1*
_output_shapes
:*
out_type0*
T0
u
gradients/Sum_1_grad/SizeConst*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
value	B :

gradients/Sum_1_grad/addAddSum_1/reduction_indicesgradients/Sum_1_grad/Size*
_class

loc:@Sum_1*
_output_shapes
: *
T0

gradients/Sum_1_grad/modFloorModgradients/Sum_1_grad/addgradients/Sum_1_grad/Size*
_class

loc:@Sum_1*
_output_shapes
: *
T0
y
gradients/Sum_1_grad/Shape_1Const*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
valueB 
|
 gradients/Sum_1_grad/range/startConst*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
value	B : 
|
 gradients/Sum_1_grad/range/deltaConst*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
value	B :
Ф
gradients/Sum_1_grad/rangeRange gradients/Sum_1_grad/range/startgradients/Sum_1_grad/Size gradients/Sum_1_grad/range/delta*
_class

loc:@Sum_1*
_output_shapes
:*

Tidx0
{
gradients/Sum_1_grad/Fill/valueConst*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
value	B :

gradients/Sum_1_grad/FillFillgradients/Sum_1_grad/Shape_1gradients/Sum_1_grad/Fill/value*
_class

loc:@Sum_1*
_output_shapes
: *
T0
ё
"gradients/Sum_1_grad/DynamicStitchDynamicStitchgradients/Sum_1_grad/rangegradients/Sum_1_grad/modgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Fill*
_class

loc:@Sum_1*#
_output_shapes
:џџџџџџџџџ*
N*
T0
z
gradients/Sum_1_grad/Maximum/yConst*
_class

loc:@Sum_1*
_output_shapes
: *
dtype0*
value	B :
Г
gradients/Sum_1_grad/MaximumMaximum"gradients/Sum_1_grad/DynamicStitchgradients/Sum_1_grad/Maximum/y*
_class

loc:@Sum_1*#
_output_shapes
:џџџџџџџџџ*
T0
Ђ
gradients/Sum_1_grad/floordivFloorDivgradients/Sum_1_grad/Shapegradients/Sum_1_grad/Maximum*
_class

loc:@Sum_1*
_output_shapes
:*
T0
Ў
gradients/Sum_1_grad/ReshapeReshapegradients/Neg_grad/Neg"gradients/Sum_1_grad/DynamicStitch*
_class

loc:@Sum_1*
_output_shapes
:*
Tshape0*
T0
М
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/floordiv*
_class

loc:@Sum_1*

Tmultiples0*'
_output_shapes
:џџџџџџџџџ*
T0
~
gradients/mul_grad/ShapeShapedense_4_target*
_class

loc:@mul*
_output_shapes
:*
out_type0*
T0
u
gradients/mul_grad/Shape_1ShapeLog*
_class

loc:@mul*
_output_shapes
:*
out_type0*
T0
Ь
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
_class

loc:@mul*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

gradients/mul_grad/mulMulgradients/Sum_1_grad/TileLog*
_class

loc:@mul*'
_output_shapes
:џџџџџџџџџ*
T0
З
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_class

loc:@mul*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
И
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
_class

loc:@mul*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Tshape0*
T0

gradients/mul_grad/mul_1Muldense_4_targetgradients/Sum_1_grad/Tile*
_class

loc:@mul*'
_output_shapes
:џџџџџџџџџ*
T0
Н
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_class

loc:@mul*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Е
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
_class

loc:@mul*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
Ѓ
gradients/Log_grad/Reciprocal
Reciprocalclip_by_value^gradients/mul_grad/Reshape_1*
_class

loc:@Log*'
_output_shapes
:џџџџџџџџџ*
T0
Є
gradients/Log_grad/mulMulgradients/mul_grad/Reshape_1gradients/Log_grad/Reciprocal*
_class

loc:@Log*'
_output_shapes
:џџџџџџџџџ*
T0

"gradients/clip_by_value_grad/ShapeShapeclip_by_value/Minimum* 
_class
loc:@clip_by_value*
_output_shapes
:*
out_type0*
T0

$gradients/clip_by_value_grad/Shape_1Const* 
_class
loc:@clip_by_value*
_output_shapes
: *
dtype0*
valueB 

$gradients/clip_by_value_grad/Shape_2Shapegradients/Log_grad/mul* 
_class
loc:@clip_by_value*
_output_shapes
:*
out_type0*
T0

(gradients/clip_by_value_grad/zeros/ConstConst* 
_class
loc:@clip_by_value*
_output_shapes
: *
dtype0*
valueB
 *    
Ю
"gradients/clip_by_value_grad/zerosFill$gradients/clip_by_value_grad/Shape_2(gradients/clip_by_value_grad/zeros/Const* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
Ћ
)gradients/clip_by_value_grad/GreaterEqualGreaterEqualclip_by_value/MinimumConst* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
є
2gradients/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/clip_by_value_grad/Shape$gradients/clip_by_value_grad/Shape_1* 
_class
loc:@clip_by_value*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
ш
#gradients/clip_by_value_grad/SelectSelect)gradients/clip_by_value_grad/GreaterEqualgradients/Log_grad/mul"gradients/clip_by_value_grad/zeros* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
Ћ
'gradients/clip_by_value_grad/LogicalNot
LogicalNot)gradients/clip_by_value_grad/GreaterEqual* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ
ш
%gradients/clip_by_value_grad/Select_1Select'gradients/clip_by_value_grad/LogicalNotgradients/Log_grad/mul"gradients/clip_by_value_grad/zeros* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ*
T0
т
 gradients/clip_by_value_grad/SumSum#gradients/clip_by_value_grad/Select2gradients/clip_by_value_grad/BroadcastGradientArgs* 
_class
loc:@clip_by_value*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
з
$gradients/clip_by_value_grad/ReshapeReshape gradients/clip_by_value_grad/Sum"gradients/clip_by_value_grad/Shape* 
_class
loc:@clip_by_value*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
ш
"gradients/clip_by_value_grad/Sum_1Sum%gradients/clip_by_value_grad/Select_14gradients/clip_by_value_grad/BroadcastGradientArgs:1* 
_class
loc:@clip_by_value*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Ь
&gradients/clip_by_value_grad/Reshape_1Reshape"gradients/clip_by_value_grad/Sum_1$gradients/clip_by_value_grad/Shape_1* 
_class
loc:@clip_by_value*
_output_shapes
: *
Tshape0*
T0

*gradients/clip_by_value/Minimum_grad/ShapeShapetruediv*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
:*
out_type0*
T0

,gradients/clip_by_value/Minimum_grad/Shape_1Const*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
: *
dtype0*
valueB 
К
,gradients/clip_by_value/Minimum_grad/Shape_2Shape$gradients/clip_by_value_grad/Reshape*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
:*
out_type0*
T0

0gradients/clip_by_value/Minimum_grad/zeros/ConstConst*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
: *
dtype0*
valueB
 *    
ю
*gradients/clip_by_value/Minimum_grad/zerosFill,gradients/clip_by_value/Minimum_grad/Shape_20gradients/clip_by_value/Minimum_grad/zeros/Const*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0
Ѕ
.gradients/clip_by_value/Minimum_grad/LessEqual	LessEqualtruedivsub*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0

:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/clip_by_value/Minimum_grad/Shape,gradients/clip_by_value/Minimum_grad/Shape_1*(
_class
loc:@clip_by_value/Minimum*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0

+gradients/clip_by_value/Minimum_grad/SelectSelect.gradients/clip_by_value/Minimum_grad/LessEqual$gradients/clip_by_value_grad/Reshape*gradients/clip_by_value/Minimum_grad/zeros*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0
Р
/gradients/clip_by_value/Minimum_grad/LogicalNot
LogicalNot.gradients/clip_by_value/Minimum_grad/LessEqual*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ

-gradients/clip_by_value/Minimum_grad/Select_1Select/gradients/clip_by_value/Minimum_grad/LogicalNot$gradients/clip_by_value_grad/Reshape*gradients/clip_by_value/Minimum_grad/zeros*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
T0

(gradients/clip_by_value/Minimum_grad/SumSum+gradients/clip_by_value/Minimum_grad/Select:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs*(
_class
loc:@clip_by_value/Minimum*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
ї
,gradients/clip_by_value/Minimum_grad/ReshapeReshape(gradients/clip_by_value/Minimum_grad/Sum*gradients/clip_by_value/Minimum_grad/Shape*(
_class
loc:@clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0

*gradients/clip_by_value/Minimum_grad/Sum_1Sum-gradients/clip_by_value/Minimum_grad/Select_1<gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*(
_class
loc:@clip_by_value/Minimum*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
ь
.gradients/clip_by_value/Minimum_grad/Reshape_1Reshape*gradients/clip_by_value/Minimum_grad/Sum_1,gradients/clip_by_value/Minimum_grad/Shape_1*(
_class
loc:@clip_by_value/Minimum*
_output_shapes
: *
Tshape0*
T0

gradients/truediv_grad/ShapeShapedense_4/Softmax*
_class
loc:@truediv*
_output_shapes
:*
out_type0*
T0
}
gradients/truediv_grad/Shape_1ShapeSum*
_class
loc:@truediv*
_output_shapes
:*
out_type0*
T0
м
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
_class
loc:@truediv*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ*
T0
Њ
gradients/truediv_grad/RealDivRealDiv,gradients/clip_by_value/Minimum_grad/ReshapeSum*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
T0
Ы
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*
_class
loc:@truediv*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
П
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0

gradients/truediv_grad/NegNegdense_4/Softmax*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
T0

 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegSum*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
T0
 
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Sum*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
T0
П
gradients/truediv_grad/mulMul,gradients/clip_by_value/Minimum_grad/Reshape gradients/truediv_grad/RealDiv_2*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
T0
Ы
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*
_class
loc:@truediv*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Х
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0

gradients/Sum_grad/ShapeShapedense_4/Softmax*
_class

loc:@Sum*
_output_shapes
:*
out_type0*
T0
q
gradients/Sum_grad/SizeConst*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
value	B :

gradients/Sum_grad/addAddSum/reduction_indicesgradients/Sum_grad/Size*
_class

loc:@Sum*
_output_shapes
: *
T0

gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
_class

loc:@Sum*
_output_shapes
: *
T0
u
gradients/Sum_grad/Shape_1Const*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
valueB 
x
gradients/Sum_grad/range/startConst*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
value	B : 
x
gradients/Sum_grad/range/deltaConst*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
value	B :
К
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*
_class

loc:@Sum*
_output_shapes
:*

Tidx0
w
gradients/Sum_grad/Fill/valueConst*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
value	B :

gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
_class

loc:@Sum*
_output_shapes
: *
T0
х
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
_class

loc:@Sum*#
_output_shapes
:џџџџџџџџџ*
N*
T0
v
gradients/Sum_grad/Maximum/yConst*
_class

loc:@Sum*
_output_shapes
: *
dtype0*
value	B :
Ћ
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
_class

loc:@Sum*#
_output_shapes
:џџџџџџџџџ*
T0

gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
_class

loc:@Sum*
_output_shapes
:*
T0
В
gradients/Sum_grad/ReshapeReshape gradients/truediv_grad/Reshape_1 gradients/Sum_grad/DynamicStitch*
_class

loc:@Sum*
_output_shapes
:*
Tshape0*
T0
Д
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*
_class

loc:@Sum*

Tmultiples0*'
_output_shapes
:џџџџџџџџџ*
T0
І
gradients/AddNAddNgradients/truediv_grad/Reshapegradients/Sum_grad/Tile*
_class
loc:@truediv*'
_output_shapes
:џџџџџџџџџ*
N*
T0
 
"gradients/dense_4/Softmax_grad/mulMulgradients/AddNdense_4/Softmax*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0
Ђ
4gradients/dense_4/Softmax_grad/Sum/reduction_indicesConst*"
_class
loc:@dense_4/Softmax*
_output_shapes
:*
dtype0*
valueB:
ђ
"gradients/dense_4/Softmax_grad/SumSum"gradients/dense_4/Softmax_grad/mul4gradients/dense_4/Softmax_grad/Sum/reduction_indices*"
_class
loc:@dense_4/Softmax*
	keep_dims( *#
_output_shapes
:џџџџџџџџџ*

Tidx0*
T0
Ё
,gradients/dense_4/Softmax_grad/Reshape/shapeConst*"
_class
loc:@dense_4/Softmax*
_output_shapes
:*
dtype0*
valueB"џџџџ   
ч
&gradients/dense_4/Softmax_grad/ReshapeReshape"gradients/dense_4/Softmax_grad/Sum,gradients/dense_4/Softmax_grad/Reshape/shape*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:џџџџџџџџџ*
Tshape0*
T0
З
"gradients/dense_4/Softmax_grad/subSubgradients/AddN&gradients/dense_4/Softmax_grad/Reshape*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0
Ж
$gradients/dense_4/Softmax_grad/mul_1Mul"gradients/dense_4/Softmax_grad/subdense_4/Softmax*"
_class
loc:@dense_4/Softmax*'
_output_shapes
:џџџџџџџџџ*
T0
П
*gradients/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_4/Softmax_grad/mul_1*"
_class
loc:@dense_4/BiasAdd*
_output_shapes
:*
data_formatNHWC*
T0
ф
$gradients/dense_4/MatMul_grad/MatMulMatMul$gradients/dense_4/Softmax_grad/mul_1dense_4/kernel/read*!
_class
loc:@dense_4/MatMul*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ж
&gradients/dense_4/MatMul_grad/MatMul_1MatMuldense_3/Tanh$gradients/dense_4/Softmax_grad/mul_1*!
_class
loc:@dense_4/MatMul*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
З
$gradients/dense_3/Tanh_grad/TanhGradTanhGraddense_3/Tanh$gradients/dense_4/MatMul_grad/MatMul*
_class
loc:@dense_3/Tanh*'
_output_shapes
:џџџџџџџџџ*
T0
П
*gradients/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_3/Tanh_grad/TanhGrad*"
_class
loc:@dense_3/BiasAdd*
_output_shapes
:*
data_formatNHWC*
T0
ф
$gradients/dense_3/MatMul_grad/MatMulMatMul$gradients/dense_3/Tanh_grad/TanhGraddense_3/kernel/read*!
_class
loc:@dense_3/MatMul*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ж
&gradients/dense_3/MatMul_grad/MatMul_1MatMuldense_2/Tanh$gradients/dense_3/Tanh_grad/TanhGrad*!
_class
loc:@dense_3/MatMul*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
З
$gradients/dense_2/Tanh_grad/TanhGradTanhGraddense_2/Tanh$gradients/dense_3/MatMul_grad/MatMul*
_class
loc:@dense_2/Tanh*'
_output_shapes
:џџџџџџџџџ*
T0
П
*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_2/Tanh_grad/TanhGrad*"
_class
loc:@dense_2/BiasAdd*
_output_shapes
:*
data_formatNHWC*
T0
ф
$gradients/dense_2/MatMul_grad/MatMulMatMul$gradients/dense_2/Tanh_grad/TanhGraddense_2/kernel/read*!
_class
loc:@dense_2/MatMul*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
ж
&gradients/dense_2/MatMul_grad/MatMul_1MatMuldense_1/Tanh$gradients/dense_2/Tanh_grad/TanhGrad*!
_class
loc:@dense_2/MatMul*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
З
$gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanh$gradients/dense_2/MatMul_grad/MatMul*
_class
loc:@dense_1/Tanh*'
_output_shapes
:џџџџџџџџџ*
T0
П
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_1/Tanh_grad/TanhGrad*"
_class
loc:@dense_1/BiasAdd*
_output_shapes
:*
data_formatNHWC*
T0
ф
$gradients/dense_1/MatMul_grad/MatMulMatMul$gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*!
_class
loc:@dense_1/MatMul*'
_output_shapes
:џџџџџџџџџ*
transpose_a( *
transpose_b(*
T0
б
&gradients/dense_1/MatMul_grad/MatMul_1MatMulinput_1$gradients/dense_1/Tanh_grad/TanhGrad*!
_class
loc:@dense_1/MatMul*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
T
Const_4Const*
_output_shapes
:*
dtype0*
valueB*    
t
Variable
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 

Variable/AssignAssignVariableConst_4*
use_locking(*
validate_shape(*
_class
loc:@Variable*
_output_shapes
:*
T0
e
Variable/readIdentityVariable*
_class
loc:@Variable*
_output_shapes
:*
T0
\
Const_5Const*
_output_shapes

:*
dtype0*
valueB*    
~

Variable_1
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
Ё
Variable_1/AssignAssign
Variable_1Const_5*
use_locking(*
validate_shape(*
_class
loc:@Variable_1*
_output_shapes

:*
T0
o
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
_output_shapes

:*
T0
T
Const_6Const*
_output_shapes
:*
dtype0*
valueB*    
v

Variable_2
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 

Variable_2/AssignAssign
Variable_2Const_6*
use_locking(*
validate_shape(*
_class
loc:@Variable_2*
_output_shapes
:*
T0
k
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
_output_shapes
:*
T0
\
Const_7Const*
_output_shapes

:*
dtype0*
valueB*    
~

Variable_3
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
Ё
Variable_3/AssignAssign
Variable_3Const_7*
use_locking(*
validate_shape(*
_class
loc:@Variable_3*
_output_shapes

:*
T0
o
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
_output_shapes

:*
T0
T
Const_8Const*
_output_shapes
:*
dtype0*
valueB*    
v

Variable_4
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 

Variable_4/AssignAssign
Variable_4Const_8*
use_locking(*
validate_shape(*
_class
loc:@Variable_4*
_output_shapes
:*
T0
k
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
_output_shapes
:*
T0
\
Const_9Const*
_output_shapes

:*
dtype0*
valueB*    
~

Variable_5
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
Ё
Variable_5/AssignAssign
Variable_5Const_9*
use_locking(*
validate_shape(*
_class
loc:@Variable_5*
_output_shapes

:*
T0
o
Variable_5/readIdentity
Variable_5*
_class
loc:@Variable_5*
_output_shapes

:*
T0
U
Const_10Const*
_output_shapes
:*
dtype0*
valueB*    
v

Variable_6
VariableV2*
_output_shapes
:*
shape:*
	container *
dtype0*
shared_name 

Variable_6/AssignAssign
Variable_6Const_10*
use_locking(*
validate_shape(*
_class
loc:@Variable_6*
_output_shapes
:*
T0
k
Variable_6/readIdentity
Variable_6*
_class
loc:@Variable_6*
_output_shapes
:*
T0
]
Const_11Const*
_output_shapes

:*
dtype0*
valueB*    
~

Variable_7
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name 
Ђ
Variable_7/AssignAssign
Variable_7Const_11*
use_locking(*
validate_shape(*
_class
loc:@Variable_7*
_output_shapes

:*
T0
o
Variable_7/readIdentity
Variable_7*
_class
loc:@Variable_7*
_output_shapes

:*
T0
J
mul_3Mulrho/readVariable/read*
_output_shapes
:*
T0
L
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@
sub_1Subsub_1/xrho/read*
_output_shapes
: *
T0
a
SquareSquare*gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
@
mul_4Mulsub_1Square*
_output_shapes
:*
T0
=
addAddmul_3mul_4*
_output_shapes
:*
T0

AssignAssignVariableadd*
use_locking(*
validate_shape(*
_class
loc:@Variable*
_output_shapes
:*
T0
f
mul_5Mullr/read*gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
M
Const_12Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_13Const*
_output_shapes
: *
dtype0*
valueB
 *  
V
clip_by_value_1/MinimumMinimumaddConst_13*
_output_shapes
:*
T0
b
clip_by_value_1Maximumclip_by_value_1/MinimumConst_12*
_output_shapes
:*
T0
B
SqrtSqrtclip_by_value_1*
_output_shapes
:*
T0
L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
@
add_1AddSqrtadd_1/y*
_output_shapes
:*
T0
G
	truediv_2RealDivmul_5add_1*
_output_shapes
:*
T0
O
sub_2Subdense_1/bias/read	truediv_2*
_output_shapes
:*
T0

Assign_1Assigndense_1/biassub_2*
use_locking(*
validate_shape(*
_class
loc:@dense_1/bias*
_output_shapes
:*
T0
P
mul_6Mulrho/readVariable_1/read*
_output_shapes

:*
T0
L
sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@
sub_3Subsub_3/xrho/read*
_output_shapes
: *
T0
c
Square_1Square&gradients/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
F
mul_7Mulsub_3Square_1*
_output_shapes

:*
T0
C
add_2Addmul_6mul_7*
_output_shapes

:*
T0

Assign_2Assign
Variable_1add_2*
use_locking(*
validate_shape(*
_class
loc:@Variable_1*
_output_shapes

:*
T0
f
mul_8Mullr/read&gradients/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
M
Const_14Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_15Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_2/MinimumMinimumadd_2Const_15*
_output_shapes

:*
T0
f
clip_by_value_2Maximumclip_by_value_2/MinimumConst_14*
_output_shapes

:*
T0
H
Sqrt_1Sqrtclip_by_value_2*
_output_shapes

:*
T0
L
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
F
add_3AddSqrt_1add_3/y*
_output_shapes

:*
T0
K
	truediv_3RealDivmul_8add_3*
_output_shapes

:*
T0
U
sub_4Subdense_1/kernel/read	truediv_3*
_output_shapes

:*
T0

Assign_3Assigndense_1/kernelsub_4*
use_locking(*
validate_shape(*!
_class
loc:@dense_1/kernel*
_output_shapes

:*
T0
L
mul_9Mulrho/readVariable_2/read*
_output_shapes
:*
T0
L
sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@
sub_5Subsub_5/xrho/read*
_output_shapes
: *
T0
c
Square_2Square*gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
C
mul_10Mulsub_5Square_2*
_output_shapes
:*
T0
@
add_4Addmul_9mul_10*
_output_shapes
:*
T0

Assign_4Assign
Variable_2add_4*
use_locking(*
validate_shape(*
_class
loc:@Variable_2*
_output_shapes
:*
T0
g
mul_11Mullr/read*gradients/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
M
Const_16Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_17Const*
_output_shapes
: *
dtype0*
valueB
 *  
X
clip_by_value_3/MinimumMinimumadd_4Const_17*
_output_shapes
:*
T0
b
clip_by_value_3Maximumclip_by_value_3/MinimumConst_16*
_output_shapes
:*
T0
D
Sqrt_2Sqrtclip_by_value_3*
_output_shapes
:*
T0
L
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
B
add_5AddSqrt_2add_5/y*
_output_shapes
:*
T0
H
	truediv_4RealDivmul_11add_5*
_output_shapes
:*
T0
O
sub_6Subdense_2/bias/read	truediv_4*
_output_shapes
:*
T0

Assign_5Assigndense_2/biassub_6*
use_locking(*
validate_shape(*
_class
loc:@dense_2/bias*
_output_shapes
:*
T0
Q
mul_12Mulrho/readVariable_3/read*
_output_shapes

:*
T0
L
sub_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@
sub_7Subsub_7/xrho/read*
_output_shapes
: *
T0
c
Square_3Square&gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
G
mul_13Mulsub_7Square_3*
_output_shapes

:*
T0
E
add_6Addmul_12mul_13*
_output_shapes

:*
T0

Assign_6Assign
Variable_3add_6*
use_locking(*
validate_shape(*
_class
loc:@Variable_3*
_output_shapes

:*
T0
g
mul_14Mullr/read&gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
M
Const_18Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_19Const*
_output_shapes
: *
dtype0*
valueB
 *  
\
clip_by_value_4/MinimumMinimumadd_6Const_19*
_output_shapes

:*
T0
f
clip_by_value_4Maximumclip_by_value_4/MinimumConst_18*
_output_shapes

:*
T0
H
Sqrt_3Sqrtclip_by_value_4*
_output_shapes

:*
T0
L
add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
F
add_7AddSqrt_3add_7/y*
_output_shapes

:*
T0
L
	truediv_5RealDivmul_14add_7*
_output_shapes

:*
T0
U
sub_8Subdense_2/kernel/read	truediv_5*
_output_shapes

:*
T0

Assign_7Assigndense_2/kernelsub_8*
use_locking(*
validate_shape(*!
_class
loc:@dense_2/kernel*
_output_shapes

:*
T0
M
mul_15Mulrho/readVariable_4/read*
_output_shapes
:*
T0
L
sub_9/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
@
sub_9Subsub_9/xrho/read*
_output_shapes
: *
T0
c
Square_4Square*gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
C
mul_16Mulsub_9Square_4*
_output_shapes
:*
T0
A
add_8Addmul_15mul_16*
_output_shapes
:*
T0

Assign_8Assign
Variable_4add_8*
use_locking(*
validate_shape(*
_class
loc:@Variable_4*
_output_shapes
:*
T0
g
mul_17Mullr/read*gradients/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
M
Const_20Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_21Const*
_output_shapes
: *
dtype0*
valueB
 *  
X
clip_by_value_5/MinimumMinimumadd_8Const_21*
_output_shapes
:*
T0
b
clip_by_value_5Maximumclip_by_value_5/MinimumConst_20*
_output_shapes
:*
T0
D
Sqrt_4Sqrtclip_by_value_5*
_output_shapes
:*
T0
L
add_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
B
add_9AddSqrt_4add_9/y*
_output_shapes
:*
T0
H
	truediv_6RealDivmul_17add_9*
_output_shapes
:*
T0
P
sub_10Subdense_3/bias/read	truediv_6*
_output_shapes
:*
T0

Assign_9Assigndense_3/biassub_10*
use_locking(*
validate_shape(*
_class
loc:@dense_3/bias*
_output_shapes
:*
T0
Q
mul_18Mulrho/readVariable_5/read*
_output_shapes

:*
T0
M
sub_11/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
B
sub_11Subsub_11/xrho/read*
_output_shapes
: *
T0
c
Square_5Square&gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
H
mul_19Mulsub_11Square_5*
_output_shapes

:*
T0
F
add_10Addmul_18mul_19*
_output_shapes

:*
T0

	Assign_10Assign
Variable_5add_10*
use_locking(*
validate_shape(*
_class
loc:@Variable_5*
_output_shapes

:*
T0
g
mul_20Mullr/read&gradients/dense_3/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
M
Const_22Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_23Const*
_output_shapes
: *
dtype0*
valueB
 *  
]
clip_by_value_6/MinimumMinimumadd_10Const_23*
_output_shapes

:*
T0
f
clip_by_value_6Maximumclip_by_value_6/MinimumConst_22*
_output_shapes

:*
T0
H
Sqrt_5Sqrtclip_by_value_6*
_output_shapes

:*
T0
M
add_11/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
H
add_11AddSqrt_5add_11/y*
_output_shapes

:*
T0
M
	truediv_7RealDivmul_20add_11*
_output_shapes

:*
T0
V
sub_12Subdense_3/kernel/read	truediv_7*
_output_shapes

:*
T0
 
	Assign_11Assigndense_3/kernelsub_12*
use_locking(*
validate_shape(*!
_class
loc:@dense_3/kernel*
_output_shapes

:*
T0
M
mul_21Mulrho/readVariable_6/read*
_output_shapes
:*
T0
M
sub_13/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
B
sub_13Subsub_13/xrho/read*
_output_shapes
: *
T0
c
Square_6Square*gradients/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
D
mul_22Mulsub_13Square_6*
_output_shapes
:*
T0
B
add_12Addmul_21mul_22*
_output_shapes
:*
T0

	Assign_12Assign
Variable_6add_12*
use_locking(*
validate_shape(*
_class
loc:@Variable_6*
_output_shapes
:*
T0
g
mul_23Mullr/read*gradients/dense_4/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
M
Const_24Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_25Const*
_output_shapes
: *
dtype0*
valueB
 *  
Y
clip_by_value_7/MinimumMinimumadd_12Const_25*
_output_shapes
:*
T0
b
clip_by_value_7Maximumclip_by_value_7/MinimumConst_24*
_output_shapes
:*
T0
D
Sqrt_6Sqrtclip_by_value_7*
_output_shapes
:*
T0
M
add_13/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
D
add_13AddSqrt_6add_13/y*
_output_shapes
:*
T0
I
	truediv_8RealDivmul_23add_13*
_output_shapes
:*
T0
P
sub_14Subdense_4/bias/read	truediv_8*
_output_shapes
:*
T0

	Assign_13Assigndense_4/biassub_14*
use_locking(*
validate_shape(*
_class
loc:@dense_4/bias*
_output_shapes
:*
T0
Q
mul_24Mulrho/readVariable_7/read*
_output_shapes

:*
T0
M
sub_15/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
B
sub_15Subsub_15/xrho/read*
_output_shapes
: *
T0
c
Square_7Square&gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
H
mul_25Mulsub_15Square_7*
_output_shapes

:*
T0
F
add_14Addmul_24mul_25*
_output_shapes

:*
T0

	Assign_14Assign
Variable_7add_14*
use_locking(*
validate_shape(*
_class
loc:@Variable_7*
_output_shapes

:*
T0
g
mul_26Mullr/read&gradients/dense_4/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
M
Const_26Const*
_output_shapes
: *
dtype0*
valueB
 *    
M
Const_27Const*
_output_shapes
: *
dtype0*
valueB
 *  
]
clip_by_value_8/MinimumMinimumadd_14Const_27*
_output_shapes

:*
T0
f
clip_by_value_8Maximumclip_by_value_8/MinimumConst_26*
_output_shapes

:*
T0
H
Sqrt_7Sqrtclip_by_value_8*
_output_shapes

:*
T0
M
add_15/yConst*
_output_shapes
: *
dtype0*
valueB
 *wЬ+2
H
add_15AddSqrt_7add_15/y*
_output_shapes

:*
T0
M
	truediv_9RealDivmul_26add_15*
_output_shapes

:*
T0
V
sub_16Subdense_4/kernel/read	truediv_9*
_output_shapes

:*
T0
 
	Assign_15Assigndense_4/kernelsub_16*
use_locking(*
validate_shape(*!
_class
loc:@dense_4/kernel*
_output_shapes

:*
T0
й
group_deps_1NoOp^mul_2^Mean_3^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15

initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign^dense_2/kernel/Assign^dense_2/bias/Assign^dense_3/kernel/Assign^dense_3/bias/Assign^dense_4/kernel/Assign^dense_4/bias/Assign
^lr/Assign^rho/Assign^decay/Assign^iterations/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign""Я
	variablesСО
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0
@
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:0
:
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0
@
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:0
:
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:0
@
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:0
:
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:0

lr:0	lr/Assign	lr/read:0

rho:0
rho/Assign
rho/read:0
%
decay:0decay/Assigndecay/read:0
4
iterations:0iterations/Assigniterations/read:0
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0
4
Variable_4:0Variable_4/AssignVariable_4/read:0
4
Variable_5:0Variable_5/AssignVariable_5/read:0
4
Variable_6:0Variable_6/AssignVariable_6/read:0
4
Variable_7:0Variable_7/AssignVariable_7/read:0"й
trainable_variablesСО
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0
@
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:0
:
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:0
@
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:0
:
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:0
@
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:0
:
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:0

lr:0	lr/Assign	lr/read:0

rho:0
rho/Assign
rho/read:0
%
decay:0decay/Assigndecay/read:0
4
iterations:0iterations/Assigniterations/read:0
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0
4
Variable_4:0Variable_4/AssignVariable_4/read:0
4
Variable_5:0Variable_5/AssignVariable_5/read:0
4
Variable_6:0Variable_6/AssignVariable_6/read:0
4
Variable_7:0Variable_7/AssignVariable_7/read:0ФШG       ЃK"	<Р=жA*

loss,ш?eo/       	>Р=жA*

val_acc9у>,э       ШС	џ>Р=жA*

val_lossЪЂ?fЉ*       чЮј	й?Р=жA*


acc^ъЩ>Ey5ђ       и-	КЁС=жA*

loss\xz?Hймa       `/п#	ЃС=жA*

val_accr?Є
N$       йм2	фЃС=жA*

val_lossъїE?ь       ё(	ЅЄС=жA*


accеё>ЈПv       и-	{KТ=жA*

loss|,?Gњж       `/п#	ЁLТ=жA*

val_acc9у>МИЩH       йм2	rMТ=жA*

val_lossD2?pЈЕе       ё(	3NТ=жA*


accч@"?њ_5|       и-	2ЌУ=жA*

lossЗ3?r =І       `/п#	Д­У=жA*

val_acc9#?tИЩ       йм2	ЇЎУ=жA*

val_lossШl?4Тv       ё(	ЏУ=жA*


accL=?ьЏf       и-	F'Х=жA*

loss8ќ?Еoб\       `/п#	Ш(Х=жA*

val_accЧq?хВ^       йм2	Г)Х=жA*

val_loss?Чa       ё(	*Х=жA*


accч@"?_)Т       и-	Ц=жA*

lossя!?~ї       `/п#	^Ц=жA*

val_acc  @?nВ       йм2	,Ц=жA*

val_loss?X;ИЭ       ё(	ёЦ=жA*


acc­0?ыы       и-	ШШ=жA*

loss?Xђ)       `/п#		Ш=жA*

val_acc9c?YDё       йм2	l
Ш=жA*

val_lossм
ѕ>EЖпЂ       ё(	-Ш=жA*


accH4?qгkH       и-	sЩ=жA*

loss}?mcиl       `/п#	uuЩ=жA*

val_accЧq?ЕB       йм2	yvЩ=жA*

val_losscPЪ>4&с3       ё(	lwЩ=жA*


acc­А9?чќ       и-	Ы=жA*

lossjO?ўzC       `/п#	Ы=жA*

val_accUUU?ЄuА       йм2	эЫ=жA*

val_lossQпП>8AkЏ       ё(	ЧЫ=жA*


accЭD?):       и-	}АЫ=жA	*

lossUчБ>јЯpU       `/п#	ЃБЫ=жA	*

val_accу8?yў]       йм2	tВЫ=жA	*

val_lossЯё?7З7       ё(	9ГЫ=жA	*


accП]j?йЫ<       и-	TVЬ=жA
*

lossк\Ї>Є`Ь       `/п#	uWЬ=жA
*

val_accЧq\?И6G       йм2	KXЬ=жA
*

val_lossVыЭ>фdўJ       ё(	YЬ=жA
*


acc/ѕd?ЁЇд