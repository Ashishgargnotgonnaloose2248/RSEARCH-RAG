CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    year INT
);

CREATE TABLE IF NOT EXISTS citations (
    citing_paper_id TEXT,
    cited_paper_id TEXT,
    PRIMARY KEY (citing_paper_id, cited_paper_id)
);
