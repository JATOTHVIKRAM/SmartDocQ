# SmartDocQ - AI-Powered Documentation Assistant

A comprehensive AI-powered platform for smart documentation and interview assistance.

## Project Structure

```
root/
├── client/      ← React/TypeScript frontend (deploy separately)
├── server/      ← Node.js/Express + Python backend (deploy separately)
├── README.md    ← This file
└── .gitignore   ← Git ignore rules
```

## Separate Deployments

This project is designed for **separate deployments** of frontend and backend:

### Frontend Deployment (`client/`)
- **Technology**: React 18 + TypeScript + Vite
- **UI**: Tailwind CSS + Radix UI components
- **Routing**: Wouter
- **State Management**: React Query

### Backend Deployment (`server/`)
- **Technology**: Node.js/Express + Python
- **Database**: MongoDB
- **Authentication**: JWT + Passport.js
- **AI Services**: Python integration

## Getting Started

### Prerequisites
- Node.js (v18 or higher)
- Python 3.8+
- MongoDB

### Frontend Setup
```bash
cd client
npm install
npm run dev
```

### Backend Setup
```bash
cd server
npm install
pip install -r requirements.txt
npm run dev
```

## Deployment

### Frontend Deployment
1. Build the React app: `cd client && npm run build`
2. Deploy the `dist/` folder to your hosting service (Vercel, Netlify, etc.)

### Backend Deployment
1. Build the server: `cd server && npm run build`
2. Deploy to your server platform (Railway, Heroku, AWS, etc.)
3. Set environment variables for database and JWT secrets

## Features

- Smart document processing and analysis
- AI-powered interview copilot
- File upload and management
- User authentication and authorization
- Modern, responsive UI with Tailwind CSS

## License

MIT