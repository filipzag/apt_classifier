/** ATT&CK host platforms the Target OS can name (mirrors apt_classifier.platforms). */
export const PLATFORMS = ['Windows', 'Linux', 'macOS', 'ESXi'] as const

export const PLATFORM_HELP =
  'Pick every operating system the activity touched; leave empty when unknown. Techniques documented only for other operating systems are not mapped (Linux rules out Windows Scheduled Task or PowerShell), so a missing OS hides its techniques. Cloud, SaaS, identity and network-device techniques are never filtered.'
