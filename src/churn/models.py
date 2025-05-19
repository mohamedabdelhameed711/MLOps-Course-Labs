from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


def get_model(name: str, **kwargs):

    if name == "rf":
        return RandomForestClassifier(**kwargs)

    if name == "gb":
        return GradientBoostingClassifier(**kwargs)

    if name == "et":
        return ExtraTreesClassifier(**kwargs)        

    if name == "lr":
        default = dict(max_iter=1000, n_jobs=-1)
        default.update(kwargs)
        return LogisticRegression(**default)

    raise ValueError(f"Unknown model name: {name!r}")
