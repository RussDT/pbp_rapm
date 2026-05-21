import express from 'express'
import cors from 'cors'
import { readdir, readFile } from 'fs/promises'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const RESULTS_DIR = join(__dirname, '..', 'nba_pipeline', 'results')

const app = express()
app.use(cors())

// List all CSV files
app.get('/api/files', async (req, res) => {
  try {
    const files = await readdir(RESULTS_DIR)
    const csvFiles = files.filter(f => f.endsWith('.csv')).sort()
    res.json(csvFiles)
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})

// Get CSV file content
app.get('/api/files/:filename', async (req, res) => {
  try {
    const filePath = join(RESULTS_DIR, req.params.filename)
    // Basic security check
    if (!filePath.startsWith(RESULTS_DIR)) {
      return res.status(403).json({ error: 'Access denied' })
    }
    const content = await readFile(filePath, 'utf-8')
    res.type('text/csv').send(content)
  } catch (err) {
    res.status(404).json({ error: 'File not found' })
  }
})

const PORT = 3001
app.listen(PORT, () => {
  console.log(`API server running at http://localhost:${PORT}`)
  console.log(`Serving CSVs from: ${RESULTS_DIR}`)
})
