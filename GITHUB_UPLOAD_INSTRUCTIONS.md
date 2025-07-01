# GitHub'a Yükleme Talimatları

Bu projede DREAM Diffusion implementasyonunuz GitHub'a yüklemeye hazır! Aşağıdaki adımları takip edin:

## 1. GitHub'da Repository Oluşturma

1. [GitHub.com](https://github.com)'a gidin ve hesabınıza giriş yapın
2. Sağ üstteki **"+"** butonuna tıklayın ve **"New repository"** seçin
3. Repository ayarları:
   - **Repository name**: `dream-diffusion` (veya istediğiniz isim)
   - **Description**: "DREAM Diffusion implementation for face generation using PyTorch"
   - **Public** seçeneğini işaretleyin
   - **DO NOT** initialize with README (bizim README'miz var)
   - **Create repository** butonuna tıklayın

## 2. Local Repository'yi GitHub'a Bağlama

Repository oluşturduktan sonra, GitHub size bir URL verecek. Bu URL'yi kullanarak:

```bash
# Terminal/Command Prompt'ta bu klasöre gidin
cd "/mnt/d/BLG Project/BLG Project Vol 5/github"

# Remote repository ekleyin (USERNAME yerine GitHub kullanıcı adınızı yazın)
git remote add origin https://github.com/USERNAME/dream-diffusion.git

# Değişiklikleri push edin
git push -u origin main
```

## 3. GitHub Personal Access Token (Eğer Gerekirse)

Eğer şifre sorulursa ve iki faktörlü doğrulama aktifse:

1. GitHub → Settings → Developer settings → Personal access tokens
2. "Generate new token" → "repo" yetkisini seçin
3. Token'ı kopyalayın ve şifre yerine kullanın

## 4. Large Files İçin Git LFS (Opsiyonel)

Eğer model checkpoint'leri paylaşacaksanız:

```bash
# Git LFS'i yükleyin (eğer yoksa)
git lfs install

# Dosyaları ekleyin ve commit yapın
git add your_checkpoint.pt
git commit -m "Add model checkpoint"
git push
```

## 5. Repository'yi Güzelleştirme

GitHub'da repository'nize girdikten sonra:

1. **About** bölümünde:
   - Description ekleyin
   - Website: Varsa blog/demo linki
   - Topics: `diffusion-models`, `pytorch`, `celeba`, `face-generation`, `dream`

2. **README** preview'ı kontrol edin

3. **Releases** bölümünden trained model paylaşabilirsiniz

## 6. Alternatif: GitHub Desktop Kullanımı

Eğer komut satırı yerine GUI tercih ederseniz:

1. [GitHub Desktop](https://desktop.github.com/) indirin
2. File → Add Local Repository → Bu klasörü seçin
3. Publish repository butonuna tıklayın

## Önemli Notlar

- `.gitignore` dosyası dataset'leri ve büyük dosyaları otomatik exclude eder
- Private bilgiler (API keys, paths) paylaşmadığınızdan emin olun
- İlk push biraz zaman alabilir

## Sorun Giderme

**"Permission denied" hatası:**
```bash
git remote set-url origin https://USERNAME@github.com/USERNAME/dream-diffusion.git
```

**"Large files" uyarısı:**
100MB üzeri dosyalar için Git LFS kullanın veya `.gitignore`'a ekleyin.

---

✨ Tebrikler! Projeniz GitHub'da yayınlanmaya hazır!