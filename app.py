import streamlit as st
import joblib
import numpy as np

# 1. Kaydedilen Dosyaları Yükle
model = joblib.load('en_iyi_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Sayfa Yapılandırması
st.set_page_config(page_title="KTÜ EM Kariyer Rehberi", layout="wide", page_icon="🎓")
st.title("🎓 KTÜ Endüstri Mühendisliği Kariyer Rehberi")
st.markdown("---")
st.write("### Merhaba! Kariyer yolculuğunu şekillendirmek için tüm akademik ve sosyal geçmişini analiz ediyoruz.")

# Not Seçeneklerini Sayıya Çeviren Sözlük
not_map = {
    "Dersi almadım": 0,
    "Orta (CC-DC-Muaf)": 1,
    "İyi (BB-CB)": 2,
    "Çok İyi (AA-BA)": 3
}

# Form Başlangıcı
with st.form("kariyer_formu"):
    # --- AKADEMİK BİLGİLER SEKSİYONU ---
    st.subheader("📊 1. Akademik Başarı ve Ders Notları")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bg = st.selectbox("Bilgisayar Programlama", list(not_map.keys()))
        hs = st.selectbox("Hizmet Sistemleri", list(not_map.keys()))
        ybs = st.selectbox("Yönetim Bilişim Sistemleri", list(not_map.keys()))
        ie = st.selectbox("İleri Excel ile Programlama", list(not_map.keys()))
        tt = st.selectbox("Tahmin Teknikleri", list(not_map.keys()))
        
    with col2:
        upk1 = st.selectbox("Üretim Planlama ve Kontrol 1", list(not_map.keys()))
        upk2 = st.selectbox("Üretim Planlama ve Kontrol 2", list(not_map.keys()))
        ya1 = st.selectbox("Yöneylem Araştırması 1", list(not_map.keys()))
        ya2 = st.selectbox("Yöneylem Araştırması 2", list(not_map.keys()))
        bz = st.selectbox("Benzetim", list(not_map.keys()))
        
    with col3:
        ttp = st.selectbox("Tesis Tasarımı ve Planlama", list(not_map.keys()))
        ietd = st.selectbox("İş Etüdü", list(not_map.keys()))
        erg = st.selectbox("Ergonomi", list(not_map.keys()))
        ort = st.number_input("Genel Not Ortalaması (GNO)", 0.0, 4.0, 2.50, step=0.01)

    st.markdown("---")
    
    # --- KİŞİSEL VE DENEYİM BİLGİLERİ SEKSİYONU ---
    st.subheader("💼 2. Kişisel Bilgiler, Staj ve Sosyal Deneyimler")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        cinsiyet = st.selectbox("Cinsiyet", [0, 1], format_func=lambda x: "Kadın" if x==0 else "Erkek")
        ing = st.selectbox("İngilizce Seviyesi", [1, 2, 3], format_func=lambda x: ["Kötü", "Orta", "İyi"][x-1])

    with c2:
        kulup = st.selectbox("Öğrenci Kulübü Üyeliği", [0, 1], format_func=lambda x: "Hayır" if x==0 else "Evet")
        erasmus = st.selectbox("Erasmus Programına Katıldınız mı?", [0, 1], format_func=lambda x: "Hayır" if x==0 else "Evet")

    with c3:
        endustristaj = st.selectbox("Endüstri stajınızı hangi sektörde yaptınız?", [1, 2], format_func=lambda x: ["Üretim", "Hizmet"][x-1])
        gonullustaj = st.selectbox("Gönüllü staj yaptınız mı? Yaptıysanız hangi sektörde yaptınız?", options=[0, 1, 2, 3, 4], format_func=lambda x: ["Hayır, yapmadım", "Üretim", "Hizmet", "Yazılım", "Akademi/Diğer"][x])

    # TAHMİN BUTONU
    submit = st.form_submit_button("🎓 KARİYERİMİ ANALİZ ET")

# Tahmin Süreci
if submit:
    # Modelin beklediği 25 özellikli boş dizi
    input_data = np.zeros(25)
    
    # Girdileri senin düzenlediğin isimlere göre yerleştiriyoruz:
    input_data[0] = not_map[bg]
    input_data[1] = not_map[hs]
    input_data[2] = not_map[ybs]
    input_data[3] = not_map[ie]
    input_data[4] = not_map[tt]
    input_data[5] = not_map[upk1]
    input_data[6] = not_map[upk2]
    input_data[7] = not_map[ya1]
    input_data[8] = not_map[ya2]
    input_data[9] = not_map[bz]
    input_data[10] = not_map[ttp]
    input_data[11] = not_map[ietd]
    input_data[12] = not_map[erg]
    
    # 13 numaralı indeksi (varsa başka bir özellik) 0 bırakıyoruz.
    input_data[14] = cinsiyet
    input_data[15] = ort
    input_data[16] = ing
    
    # 17 ve 18 boş kalıyor (0)
    input_data[19] = kulup
    input_data[20] = erasmus
    input_data[21] = endustristaj
    input_data[22] = gonullustaj
    
    # Ölçeklendirme ve Tahmin
    try:
        input_scaled = scaler.transform([input_data])
        tahmin_no = model.predict(input_scaled)
        tahmin_isim = le.inverse_transform(tahmin_no)[0]

        # Sonuç Ekranı
        st.markdown("---")
        st.balloons()
        st.success(f"## 🎉 Tahmin Sonucu: **{tahmin_isim.upper()}**")
        st.info("Bu tahmin, KTÜ EM mezun verilerine dayanarak yapılmıştır.")
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")