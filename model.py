from darts.models import NHiTSModel
from darts.metrics import mae
from darts import TimeSeries


def train(scaled_train_data, test_data, eps=10000):
    
    nhits = NHiTSModel(input_chunk_length=len(scaled_train_data) - len(test_data), output_chunk_length=len(test_data), random_state=42)

    nhits.fit(scaled_train_data, epochs=eps)

    return nhits

def predict(model, input_data):
    scaled_pred_nhits = model.predict(n=len(input_data))
    pred_nhits = model.predict(n=120)
    pred_nhits = train_scaler.inverse_transform(scaled_pred_nhits)

    mae_nhits = mae(input_data, pred_nhits)

    # print(f'baseline: {naive_mae}, N-Hits: {mae_nhits}')
    print(f'N-Hits: {mae_nhits}')
    train.plot(label='train')
    test.plot(label='Actual')
    # pred_naive.plot(label='Baseline')
    pred_nhits.plot(label='N-HiTS')

    plt.show()