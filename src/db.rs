// database (sqlite3) reading

use rusqlite::Connection;
use rusqlite::Row;
use std::convert::TryFrom;
use serde::Serialize;


#[derive(Debug, Serialize)]
pub struct Entry {
    id: String,
    #[serde(skip_serializing)]
    pub embedding: Vec<f32>, // TODO consider arrow, ndarray ...
    content: String,
    pub score: f32,
}

fn blob2f32(bytes: &Vec<u8>) -> Result<Vec<f32>, std::io::Error> {
    use std::io::Cursor;
    use byteorder::{LittleEndian, ReadBytesExt};

    let mut cursor = Cursor::new(bytes);
    let num_f32s = cursor.get_ref().len() / 4;
    let mut f32_vec = Vec::with_capacity(num_f32s);
    for _ in 0..num_f32s {
        f32_vec.push(cursor.read_f32::<LittleEndian>()?);
    }
    Ok(f32_vec)
}

impl TryFrom<&Row<'_>> for Entry {
    type Error = rusqlite::Error;
    fn try_from(row: &Row) -> Result<Entry, Self::Error> {
        let id: String = row.get(0).unwrap();
        let raw: Vec<u8> = row.get(1).unwrap();
        let embedding: Vec<f32> = blob2f32(&raw).unwrap();
        let content: String = row.get(2).unwrap();
        let score: f32 = 0.;
        Ok(Entry { id, embedding, content, score })
    }
}

pub fn retrieve() -> Result<Vec<Entry>, Box<dyn std::error::Error>> {
    let conn = Connection::open("/home/vojta/bin/embeddings/knowledgebase.1.db")?;
    let mut stmt = conn.prepare("select id, embedding, content from embeddings")?;
    let iter = stmt.query_map([], |row| { Entry::try_from(row) })?;
    let col: Vec<Entry> = iter.map(|entry| entry.unwrap()).collect();
    Ok(col)
}
