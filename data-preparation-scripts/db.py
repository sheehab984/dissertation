import sqlalchemy as db
from sqlalchemy import text
from sqlalchemy import MetaData, Table, Column, String, Float, Integer, insert
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base

class Database():
    # replace the user, password, hostname and database according to your configuration according to your information
    engine = db.create_engine('postgresql://sheehabpranto:@localhost:5432/postgres')
    def __init__(self):
        self.connection = self.engine.connect()
        print("DB Instance created")
    def fetchByQyery(self, query):
        fetchQuery = self.connection.execute(text(f"SELECT * FROM {query}"))
        return fetchQuery.fetchall()

    def createTable(self, table_name):
        metadata = MetaData()
        metadata.reflect(bind=self.engine)

        if not self.engine.dialect.has_table(self.connection, table_name):
            # The table doesn't exist, let's create it
            table = Table(table_name, metadata,
                        Column('timestamp', String,  primary_key=True),
                        Column('bid', Float),
                        Column('ask', Float),
                        Column('vol', Integer)
                        )
            metadata.create_all(self.engine)

# Base = declarative_base()

# class TickData(Base):
#     """Model for bid data."""
#     __tablename__ = 'TickData'
#     timestamp = Column(String, primary_key=True)
#     bid = Column(Float)
#     ask = Column(Float)
    
#     def __repr__(self): 
#         return "<Tick Data(timestamp='%s', bid='%s', ask='%s')>" % (self.timestamp, self.bid, self.ask)

# if __name__ == "__main__":
#     db = Database()
#     db.fetchAllUsers()