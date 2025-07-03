from typing import Optional
from neo4j import GraphDatabase

class MemgraphDriver:
    def __init__(
            self, 
            uri: str = "bolt://localhost:7687", 
            username: str = "memgraphmikey", 
            password: str = "memgraphmikey"
        ):
      
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def test_connection(self):
        with self.driver.sesion() as session:
            result = session.run("RETURN 1")
            print(result.single()[0])
    
    def close(self):
        self._driver.close()

    def execute_query(self, query: str, parameters: Optional[dict] = None) -> list:
        """Execute a Cypher query and return the results."""
        with self._driver.session() as session:
            result = session.run(query, parameters or {})
            return [record for record in result]

    def execute_write_query(self, query: str, parameters: Optional[dict] = None) -> None:
        """Execute a write Cypher query."""
        with self._driver.session() as session:
            session.run(query, parameters or {})