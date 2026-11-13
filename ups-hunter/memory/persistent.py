import sqlite3

conexao = sqlite3.connect("ups-hunter\data\ups-hunter.db")
cursor = conexao.cursor()

cursor.execute("CREATE TABLE session")