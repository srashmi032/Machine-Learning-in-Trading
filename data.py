# Load the data

#from backtester.dataSource.quant_quest_data_source import QuantQuestDataSource
from backtester.features.feature import Feature
from backtester.trading_system import TradingSystem
from backtester.sample_scripts.fair_value_params import FairValueTradingParams
from backtester.version import updateCheck


cachedFolderName = '/Users/rashmisahu/desktop/Auquan/qq2solver-data/historicalData/'
dataSetId = 'trainingData1'
instrumentIds = ['MQK']
ds = QuantQuestDataSource(cachedFolderName=cachedFolderName,
                                    dataSetId=dataSetId,
                                    instrumentIds=instrumentIds)
def loadData(ds):
    data = None
    for key in ds.getBookDataByFeature().keys():
        if data is None:
            data = pd.DataFrame(np.nan, index = ds.getBookDataByFeature()[key].index, columns=[])
        data[key] = ds.getBookDataByFeature()[key]
    data['Stock Price'] =  ds.getBookDataByFeature()['stockTopBidPrice'] + ds.getBookDataByFeature()['stockTopAskPrice'] / 2.0
    data['Future Price'] = ds.getBookDataByFeature()['futureTopBidPrice'] + ds.getBookDataByFeature()['futureTopAskPrice'] / 2.0
    data['Y(Target)'] = ds.getBookDataByFeature()['basis'].shift(-5)
    del data['benchmark_score']
    del data['FairValue']
    return data
data = loadData(ds)