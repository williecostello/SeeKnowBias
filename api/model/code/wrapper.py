import pickle

class ModelWrapper():
  def __init__(self, model):
    self._model = model

  def predict(self, data):
    return self._model.predict(data.text)

def _load_pyfunc(path):
  # Load the model object
  with open(f'{path}/model.pkl', 'rb') as f:
    model = pickle.load(f)

  return ModelWrapper(
      model=model,
  )