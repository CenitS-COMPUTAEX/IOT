from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis

from defs.enums import ModelType
from models.base_model import BaseModel

from defs.exceptions import IllegalOperationError
from defs.config import Config as Cfg

"""
K-nearest neighbors model
"""

# If the number of samples passed to _get_trained_model() is higher than this value, only this amount of samples
# will be used to train NCA.
MAX_SAMPLES_NCA = 1500


class KnnModel(BaseModel):
	trained_model: "KNeighborsClassifier | None"
	trained_nca: "NeighborhoodComponentsAnalysis | None"

	def __init__(self, saved_data=None):
		"""
		Creates an instance of this model.
		saved_data: Trained model and NCA. If unspecified, this instance must be trained before it can be used
		for prediction.
		"""
		if saved_data is None:
			self.trained_model = None
			self.trained_nca = None
		else:
			self.trained_model = saved_data[0]
			self.trained_nca = saved_data[1]

	def get_model_type(self) -> ModelType:
		return ModelType.KNN

	def get_data_to_save(self) -> object:
		return [self.trained_model, self.trained_nca]

	def _train_model(self, x, y):
		x_nca = x
		y_nca = y
		if len(x) > MAX_SAMPLES_NCA:
			# Use a StratifiedShuffleSplit to ensure that the frequency of each class is maintained
			splitter = StratifiedShuffleSplit(n_splits=1, test_size=MAX_SAMPLES_NCA, random_state=0)
			for _, indexes in splitter.split(x, y):
				x_nca = x[indexes].copy()
				y_nca = y[indexes].copy()

		nca = NeighborhoodComponentsAnalysis(random_state=0)
		nca.fit(x_nca, y_nca)
		self.trained_nca = nca
		x_transform = self.trained_nca.transform(x)

		model = KNeighborsClassifier(n_neighbors=Cfg.get().knn_num_neighbors, weights="distance", metric="manhattan",
			algorithm="ball_tree", n_jobs=-1)
		model.fit(x_transform, y)
		self.trained_model = model

	def get_prediction(self, x):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			x_transform = self.trained_nca.transform(x)
			return self.trained_model.predict(x_transform)

	def get_multi_prediction(self, x):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before predictions can be made.")
		else:
			x_transform = self.trained_nca.transform(x)
			return self.trained_model.predict_proba(x_transform)

	def get_classes(self):
		if self.trained_model is None:
			raise IllegalOperationError("The model must be trained before classes can be returned.")
		else:
			return self.trained_model.classes_
