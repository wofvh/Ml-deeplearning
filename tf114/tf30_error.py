Exception: in user code:

    /opt/conda/lib/python3.7/site-packages/keras/engine/training.py:853 train_function  *
        return step_function(self, iterator)
    /tmp/ipykernel_34/1396363757.py:53 class_loss_regr_fixed_num  *
        x = y_true[:, :, 4*num_classes:] - y_pred
    /opt/conda/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:1383 binary_op_wrapper
        raise e
    /opt/conda/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:1367 binary_op_wrapper
        return func(x, y, name=name)
    /opt/conda/lib/python3.7/site-packages/tensorflow/python/util/dispatch.py:206 wrapper
        return target(*args, **kwargs)
    /opt/conda/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:548 subtract
        return gen_math_ops.sub(x, y, name)
    /opt/conda/lib/python3.7/site-packages/tensorflow/python/ops/gen_math_ops.py:10654 sub
        "Sub", x=x, y=y, name=name)
    /opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:558 _apply_op_helper
        inferred_from[input_arg.type_attr]))

    TypeError: Input 'y' of 'Sub' Op has type float32 that does not match type int64 of argument 'x'.

Exception: 'NoneType' object is not callable