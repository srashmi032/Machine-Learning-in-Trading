from backtester.features.feature import Feature
from backtester.trading_system import TradingSystem
from backtester.sample_scripts.fair_value_params import FairValueTradingParams
from backtester.version import updateCheck


class Problem1Solver():


    def getTrainingDataSet(self):
        return "trainingData1"

    

    def getSymbolsToTrade(self):
        return ['AGW']

    

    def getCustomFeatures(self):
        return {'my_custom_feature': MyCustomFeature}

    

    def getFeatureConfigDicts(self):
        ma2Dict = {'featureKey': 'ma_5',
                   'featureId': 'moving_average',
                   'params': {'period': 5,
                              'featureName': 'stockTopAskPrice'}}
        ma1Dict = {'featureKey': 'ma_5',
                   'featureId': 'moving_average',
                   'params': {'period': 5,
                              'featureName': 'basis'}}
        expma = {'featureKey': 'exponential_moving_average',
                 'featureId': 'exponential_moving_average',
                 'params': {'period': 20,
                              'featureName': 'basis'}}
        sdevDict = {'featureKey': 'sdev_5',
                    'featureId': 'moving_sdev',
                    'params': {'period': 5,
                               'featureName': 'basis'}}
        customFeatureDict = {'featureKey': 'custom_inst_feature',
                             'featureId': 'my_custom_feature',
                             'params': {'param1': 'value1'}}
        return [ma1Dict, sdevDict, expma]

    

    def getFairValue(self, updateNum, time, instrumentManager):
        # holder for all the instrument features
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()

        # dataframe for a historical instrument feature (ma_5 in this case). The index is the timestamps
        # atmost upto lookback data points. The columns of this dataframe are the stock symbols/instrumentIds.
        ma5Data = lookbackInstrumentFeatures.getFeatureDf('ma_5')

        # Returns a series with index as all the instrumentIds. This returns the value of the feature at the last
        # time update.
        ma5 = ma5Data.iloc[-1]

        return ma5





class MyCustomFeature(Feature):
    
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        # Custom parameter which can be used as input to computation of this feature
        param1Value = featureParams['param1']

        # A holder for the all the instrument features
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()

        # dataframe for a historical instrument feature (basis in this case). The index is the timestamps
        # atmost upto lookback data points. The columns of this dataframe are the stocks/instrumentIds.
        lookbackInstrumentBasis = lookbackInstrumentFeatures.getFeatureDf('basis')

        # The last row of the previous dataframe gives the last calculated value for that feature (basis in this case)
        # This returns a series with stocks/instrumentIds as the index.
        currentBasisValue = lookbackInstrumentBasis.iloc[-1]

        if param1Value == 'value1':
            return currentBasisValue * 0.1
        else:
            return currentBasisValue * 0.5


if __name__ == "__main__":
    if updateCheck():
        print('Your version of the auquan toolbox package is old. Please update by running the following command:')
        print('pip install -U auquan_toolbox')
    else:
        problem1Solver = Problem1Solver()
        tsParams = FairValueTradingParams(problem1Solver)
        tradingSystem = TradingSystem(tsParams)
        # Set shouldPlot to True to quickly generate csv files with all the features
        # Set onlyAnalyze to False to run a full backtest
        # Set makeInstrumentCsvs to True to make instrument specific csvs in runLogs. This degrades the performance of the backtesting system
        
        tradingSystem.startTrading(onlyAnalyze=False, shouldPlot=True, makeInstrumentCsvs=False)





        