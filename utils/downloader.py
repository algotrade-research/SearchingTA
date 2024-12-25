from typing import Callable, List, Tuple
import pg8000
import pandas as pd

class Downloader:
    def __init__(self, processor: Callable=None):
        self._info = {
            "host": "api.algotrade.vn",
            "port": 5432,
            "user": "intern_read_only",
            "password": "ZmDaLzFf8pg5"
        }

        self._processor = processor

        self._connect()
    
    def _connect(self) -> None:
        self.conn = pg8000.connect(
            host=self._info["host"],
            port=self._info["port"],
            user=self._info["user"],
            password=self._info["password"],
            database = "algotradeDB",
            )

    def get_historical_data(self, 
                  end_date: str=None, 
                  start_date: str="2023-01-01",
                  ticker: str="VN30F1M", 
                  interval: str=None) -> pd.DataFrame:
        try:
            cur = self.conn.cursor()
            
            query = """
                SELECT
                    m.datetime,
                    m.price,
                    bp.price as bid_price,
                    ap.price as ask_price,
                    v.quantity
                FROM quote.matched m
                JOIN
                    quote.futurecontractcode fc ON DATE(m.datetime) = fc.datetime AND m.tickersymbol = fc.tickersymbol
                LEFT JOIN
                    quote.total v ON v.datetime = m.datetime AND v.tickersymbol = m.tickersymbol
                LEFT JOIN
                    quote.bidprice bp ON bp.datetime = m.datetime AND bp.tickersymbol = m.tickersymbol
                LEFT JOIN
                    quote.askprice ap ON ap.datetime = m.datetime AND ap.tickersymbol = m.tickersymbol
                WHERE
                    m.datetime BETWEEN %s AND %s
                    AND fc.futurecode = %s
                    AND bp.depth=1 AND ap.depth=1
                ORDER BY m.datetime
            """
            
            # Execute a query
            cur.execute(query, (start_date, end_date, ticker))
            result = cur.fetchall()
            cur.close()

            # Return the result 
            result = pd.DataFrame(result, columns=["datetime", "price", "bid_price", "ask_price", "volume"])
            result.set_index("datetime", inplace=True)
            result.index = pd.to_datetime(result.index)
            result = result.astype(float)
            # #print(result)

            # Convert the DatetimeIndex to just the time component
            times = result.index.time

            # Create a boolean mask for times outside the range 14:30 to 09:15
            # Since the range crosses midnight, we handle it in two parts:
            # 1. Times after 14:30 on day 1
            # 2. Times before 09:15 on day 2
            mask = (times <= pd.to_datetime('14:30').time()) & (times >= pd.to_datetime('09:00').time())

            # Filter the DataFrame to keep rows outside the specified time range
            result = result[mask]

            if interval:
                result = result.resample(interval).agg({
                    "price": "ohlc",
                    "quantity": "sum"
                }).dropna()

                result.columns = result.columns.droplevel(0)

            if self._processor:
                result = self._processor(result)
            
            return result

        except Exception as e:
            #print(f"An error occurred: {e}")
            return None
        
    def query(self, query: str) -> pd.DataFrame:
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            result = cur.fetchall()
            cur.close()
            return pd.DataFrame(result)
        except Exception as e:
            #print(f"An error occurred: {e}")
            return None
    
    def close(self) -> None:
        self.conn.close()