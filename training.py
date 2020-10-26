import trax
from trax import layers as tl
from trax.supervised import training


def create_model(input_size):

    model = tl.Serial(
        tl.convolution.Conv1d(input_size, 3),
        tl.pooling.AvgPool(3, None),
        tl.attention(4)
    )
    return model


def prepare_tasks(train_stream, eval_stream):
    train_task = training.TrainTask(
        labeled_data = train_stream,
        loss_layer = tl.CrossEntropyLoss(),
        optimizer = trax.optimizers.Adam(0.01),
        lr_schedule = trax.lr.warmup_and_rsqrt_decay(1000, 0.01),
        n_steps_per_checkpoint = 10,

    )

    eval_task = training.EvalTask(

        labeled_data=eval_stream,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
    )


def train_model(Model, train_task, eval_task):
    training_loop = training.Loop(
                        Model(mode='train'),
                        train_task,
                        eval_tasks=[eval_task],
                        output_dir='output')


if __name__ == "__main__":
    Model = create_model(33)
    train_task, eval_task = prepare_tasks()
    train_model(Model, train_task, eval_task)
