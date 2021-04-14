from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

models = {
	"decision-tree-gini": DecisionTreeClassifier(
		criterion="gini",
		max_depth=5
	),
 	"decision-tree-entropy": DecisionTreeClassifier(
      criterion="entropy",
      max_depth=5
	),
	# Ensemble
	"rf": RandomForestClassifier(n_jobs=-1, max_depth=5),
	"rf-shallow": RandomForestClassifier(
     n_estimators=10000,
     max_depth=3,
     n_jobs=-1
     ),



	"adaboost": AdaBoostClassifier(),
 	"gradientboost": GradientBoostingClassifier(),
	"extratrees": ExtraTreesClassifier(),
	"vote-ensemble": VotingClassifier(n_jobs=-1, estimators=[("rf",RandomForestClassifier()),
                                                          ("et",ExtraTreesClassifier()),
                                                          ("ab",AdaBoostClassifier()),
                                                          ("gb", GradientBoostingClassifier())]),
	"svm": SVC(cache_size=1500),

	"mlp": MLPClassifier(max_iter=1000, verbose=10),
 	"mlp-opt": MLPClassifier(max_iter=500,
                           n_iter_no_change=20,
                           learning_rate="adaptive",
                           alpha=0.0001,
                           activation="relu",
                           hidden_layer_sizes=(1000,1000),
                           batch_size=64)


}