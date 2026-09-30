import { Navigate, Route, Routes } from 'react-router-dom'

import { AuthGate } from './components/AuthGate'
import { AppShell } from './components/layout/AppShell'
import { AssessmentPage } from './pages/Assessment'
import { CanvasPage } from './pages/Canvas'
import { DashboardPage } from './pages/Dashboard'
import { DiamondPage } from './pages/Diamond'
import { DistinguishPage } from './pages/Distinguish'
import { EvidencePage } from './pages/Evidence'
import { IntakePage } from './pages/Intake'
import { IntelPage } from './pages/Intel'
import { InterviewPage } from './pages/Interview'
import { ReviewPage } from './pages/Review'
import { TechniquesPage } from './pages/Techniques'
import { UsersPage } from './pages/Users'

export default function App() {
  return (
    <AuthGate>
      <Routes>
        <Route path="login" element={<Navigate to="/" replace />} />
      <Route element={<AppShell />}>
        <Route index element={<DashboardPage />} />
        <Route path="new" element={<IntakePage />} />
        <Route path="cases/:id" element={<AssessmentPage />} />
        <Route path="cases/:id/evidence" element={<EvidencePage />} />
        <Route path="cases/:id/techniques" element={<TechniquesPage />} />
        <Route path="cases/:id/distinguish" element={<DistinguishPage />} />
        <Route path="cases/:id/interview" element={<InterviewPage />} />
        <Route path="cases/:id/canvas" element={<CanvasPage />} />
        <Route path="cases/:id/diamond" element={<DiamondPage />} />
        <Route path="cases/:id/intel" element={<IntelPage />} />
        <Route path="cases/:id/review" element={<ReviewPage />} />
        <Route path="admin/users" element={<UsersPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
      </Routes>
    </AuthGate>
  )
}
