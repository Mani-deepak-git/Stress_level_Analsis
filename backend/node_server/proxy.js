const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = 8080;

app.use((req, res, next) => {
    console.log(`[PROXY] Requesting: ${req.method} ${req.url}`);
    next();
});

const onProxyError = (err, req, res, serverName) => {
    console.error(`[${serverName} Error]:`, err.message);
    if (res && !res.headersSent) {
        res.status(502).send(`${serverName} is down or unreachable. Check if it's running.`);
    }
};

app.use('/ws/voice-confidence', createProxyMiddleware({ 
    target: 'http://127.0.0.1:8002', 
    ws: true,
    changeOrigin: true,
    onError: (err, req, res) => onProxyError(err, req, res, 'Voice Server (8002)')
}));

app.use('/socket.io', createProxyMiddleware({ 
    target: 'http://127.0.0.1:3000', 
    ws: true,
    changeOrigin: true,
    onError: (err, req, res) => onProxyError(err, req, res, 'Node Server Socket (3000)')
}));

app.use('/api', createProxyMiddleware({ 
    target: 'http://127.0.0.1:3000',
    changeOrigin: true,
    onError: (err, req, res) => onProxyError(err, req, res, 'Node Server API (3000)')
}));

app.use('/', createProxyMiddleware({ 
    target: 'http://127.0.0.1:3001',
    ws: true, 
    changeOrigin: true,
    onProxyReq: (proxyReq, req, res) => { proxyReq.setHeader('Host', '127.0.0.1:3001'); },
    onError: (err, req, res) => {
        console.error('[Frontend Server Error]:', err.message);
        if (res && !res.headersSent) {
            res.status(502).send('<h2>React Frontend (3001) is not ready!</h2><p>Wait for React to finish starting up.</p>');
        }
    }
}));

app.listen(PORT, () => {
    console.log('\n====================================================');
    console.log(`🚀 NGROK UNIFIED PROXY SERVER RUNNING ON PORT ${PORT}`);
    console.log('====================================================');
});
