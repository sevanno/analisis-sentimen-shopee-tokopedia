-- Jalankan di Supabase SQL Editor sebagai role pemilik/admin (umumnya postgres).
-- Ganti placeholder password terlebih dahulu. Jangan commit file ini setelah diisi.
-- Role ini hanya boleh INSERT ke dw.dim_date dan dw.fact_review.
-- Role dashboard_reader yang sudah ada tetap dipakai untuk SELECT dashboard.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = 'dashboard_ingest'
    ) THEN
        CREATE ROLE dashboard_ingest
            LOGIN
            PASSWORD 'REPLACE_WITH_LONG_RANDOM_PASSWORD';
    ELSE
        ALTER ROLE dashboard_ingest
            WITH LOGIN
            PASSWORD 'REPLACE_WITH_LONG_RANDOM_PASSWORD';
    END IF;
END
$$;

ALTER ROLE dashboard_ingest
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;

GRANT CONNECT ON DATABASE postgres TO dashboard_ingest;
GRANT USAGE ON SCHEMA dw TO dashboard_ingest;

-- Reset privilege tabel jika role ini pernah dipakai/diatur sebelumnya.
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA dw FROM dashboard_ingest;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA dw FROM dashboard_ingest;

-- Tidak memberi SELECT, UPDATE, DELETE, TRUNCATE, atau akses staging/log.
GRANT INSERT ON TABLE dw.dim_date, dw.fact_review TO dashboard_ingest;

ALTER TABLE dw.dim_date ENABLE ROW LEVEL SECURITY;
ALTER TABLE dw.fact_review ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS dashboard_ingest_insert_dim_date ON dw.dim_date;
CREATE POLICY dashboard_ingest_insert_dim_date
    ON dw.dim_date
    FOR INSERT
    TO dashboard_ingest
    WITH CHECK (true);

DROP POLICY IF EXISTS dashboard_ingest_insert_fact_review ON dw.fact_review;
CREATE POLICY dashboard_ingest_insert_fact_review
    ON dw.fact_review
    FOR INSERT
    TO dashboard_ingest
    WITH CHECK (true);

-- Verifikasi privilege yang diberikan:
SELECT
    has_schema_privilege('dashboard_ingest', 'dw', 'USAGE') AS can_use_dw,
    has_table_privilege('dashboard_ingest', 'dw.dim_date', 'INSERT') AS can_insert_date,
    has_table_privilege('dashboard_ingest', 'dw.fact_review', 'INSERT') AS can_insert_fact,
    has_table_privilege('dashboard_ingest', 'dw.fact_review', 'SELECT') AS can_select_fact,
    has_table_privilege('dashboard_ingest', 'dw.fact_review', 'UPDATE') AS can_update_fact,
    has_table_privilege('dashboard_ingest', 'dw.fact_review', 'DELETE') AS can_delete_fact;
