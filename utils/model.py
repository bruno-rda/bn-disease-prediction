import bnlearn as bn
import pandas as pd
from pgmpy.models import BayesianModel
from pydantic import BaseModel, create_model
from typing import Optional

def save_model(model: BayesianModel, path: str) -> None:
    bn.save(model, filepath=path, overwrite=True)

def load_model(path: str) -> BayesianModel:
    return bn.load(filepath=path)

def create_pydantic_model(symptoms: list[str]) -> BaseModel:
    '''
    Creates a Pydantic model with the fields being the symptoms 
    used to train the Bayesian Network.

    A symptom is:
    - True:  the symptom is present.
    - False: the symptom is not present.
    - None:  the symptom may or may not be present.
    '''
    fields = {name: (Optional[bool], None) for name in symptoms}
    PredictionModel = create_model('PredictionModel', **fields)
    return PredictionModel

def predict(model: BayesianModel, evidence: BaseModel, y_col: str) -> dict:
    '''
    Predicts the possible diseases and their probabilities given the evidence.
    '''
    evidence_dict = {
        k: v for k, v in evidence.model_dump().items() 
        if v is not None
    }

    query = bn.inference.fit(
        model,
        variables=[y_col],
        evidence=evidence_dict,
    )

    return query.df.round(6).sort_values('p', ascending=False)

def predict_df(
    df: pd.DataFrame, 
    model: BayesianModel, 
    PredictionModel: BaseModel,
    y_col: str
) -> pd.DataFrame:
    '''
    df[y_col] is the target column (disease)
    df.drop(columns=[y_col]) are the symptoms one hot encoded
    '''
    
    def predict_row(row: pd.Series) -> pd.Series:
        evidence = PredictionModel(**row)
        pred = predict(model, evidence, y_col).iloc[0]

        return pd.Series(
            {
                'real': row[y_col],
                'pred': pred[y_col],
                'p': pred['p']
            }
        )

    return df.apply(
        predict_row,
        axis=1
    )

def get_accuracy(df: pd.DataFrame) -> float:
    '''
    df: Dataframe obtained from predict_df
    '''
    return (df['real'] == df['pred']).sum() / df.shape[0]

def get_summary(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Groups the dataframe by the true disease label ('real') and calculates the 
    mean, minimum, maximum, and count of the predicted probabilities ('p'). 
    The result is sorted by the mean probability from lowest to highest.

    df: Dataframe obtained from predict_df
    '''
    return df.groupby('real') \
        .agg({'p': ['mean', 'min', 'max', 'count']}) \
        .sort_values(('p', 'mean'), ascending=True)
