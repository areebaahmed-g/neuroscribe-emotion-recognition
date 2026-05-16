"""
NeuroScribe — CPU-Friendly Face Emotion Recognition
v4.2 — FIXED LIVE WEBCAM DETECTION
Fixed: CLAHE preprocessing, face detection, proper cropping, normalization

Run: streamlit run neuroscribe_v4_fixed.py
"""

import os, io, zipfile, tempfile, time, warnings, threading, json
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import cv2
from datetime import datetime
from collections import Counter
import csv

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore, auth as fb_auth
import pyrebase

# ══════════════════════════════════════════════════════════════════
#  FIREBASE CONFIGURATION
# ══════════════════════════════════════════════════════════════════
FIREBASE_CONFIG = {
    "apiKey":            "AIzaSyDivQERur2WPGD5bB5d9W6IhfksMNLJ_tQ",
    "authDomain":        "neuroscribe-7c818.firebaseapp.com",
    "projectId":         "neuroscribe-7c818",
    "storageBucket":     "neuroscribe-7c818.firebasestorage.app",
    "messagingSenderId": "820189087988",
    "appId":             "1:820189087988:web:7b00dc1d5be03f420e36a1",
    "databaseURL":       ""
}

SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
CLASSES   = ['angry','contempt','disgust','fear','happy','neutral','sad','surprise']
EMOJIS    = {'angry':'😠','contempt':'😒','disgust':'🤢','fear':'😨',
             'happy':'😄','neutral':'😐','sad':'😢','surprise':'😲'}
COLORS    = {'angry':'#d44','contempt':'#b84','disgust':'#6a6','fear':'#84a',
             'happy':'#cc0','neutral':'#68a','sad':'#68b','surprise':'#c69'}
IMG_SIZE   = 48
MODEL_PATH = "neuroscribe_model.pth"
EMA_ALPHA  = 0.25
FACE_SCALE      = 1.1
FACE_NEIGHBORS  = 3
FACE_MIN_SIZE   = 60
FACE_PAD        = 0.20

ADMIN_PAGES = ["Overview","Dataset Upload","Train","Inference",
               "Live Webcam","Test Images","Evaluation","Admin Panel"]
USER_PAGES  = ["Overview","Inference","Live Webcam"]

# ══════════════════════════════════════════════════════════════════
#  Firebase initialisation
# ══════════════════════════════════════════════════════════════════
_fb_ready  = False
_db        = None
_pb_auth   = None

def _init_firebase():
    """Initialise firebase-admin + pyrebase once."""
    global _fb_ready, _db, _pb_auth
    if _fb_ready:
        return True
    try:
        if not firebase_admin._apps:
            if not os.path.exists(SERVICE_ACCOUNT_PATH):
                return False
            cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred)

        _db = firestore.client()
        pb = pyrebase.initialize_app(FIREBASE_CONFIG)
        _pb_auth = pb.auth()
        _fb_ready = True
        return True
    except Exception as e:
        print(f"Firebase init error: {e}")
        return False

def fb_sign_up(email: str, password: str, role: str = "user"):
    """Create new user account."""
    if not _init_firebase():
        return None, "Firebase not configured."
    try:
        user = fb_auth.create_user(email=email, password=password)
        _db.collection("users").document(user.uid).set({
            "email": email,
            "role": role,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "status": "active"
        })
        sign_in_user = _pb_auth.sign_in_with_email_and_password(email, password)
        return sign_in_user, None
    except Exception as e:
        msg = str(e)
        if "EMAIL_EXISTS" in msg:
            return None, "Email already exists."
        if "WEAK_PASSWORD" in msg:
            return None, "Password too weak. Use at least 6 characters."
        return None, f"Sign-up error: {msg[:120]}"

def fb_sign_in(email: str, password: str):
    """Sign in with email/password."""
    if not _init_firebase():
        return None, "Firebase not configured."
    try:
        user = _pb_auth.sign_in_with_email_and_password(email, password)
        return user, None
    except Exception as e:
        msg = str(e)
        if "INVALID_PASSWORD" in msg or "EMAIL_NOT_FOUND" in msg:
            return None, "Invalid email or password."
        if "TOO_MANY_ATTEMPTS" in msg:
            return None, "Too many attempts. Try again later."
        return None, f"Login error: {msg[:120]}"

def fb_get_role(uid: str) -> str:
    """Fetch user role from Firestore."""
    if not _fb_ready or _db is None:
        return "user"
    try:
        doc = _db.collection("users").document(uid).get()
        if doc.exists:
            return doc.to_dict().get("role", "user")
        return "user"
    except Exception:
        return "user"

def fb_log_prediction(uid: str, email: str, emotion: str, confidence: float):
    """Log a prediction to Firestore."""
    if not _fb_ready or _db is None:
        return
    try:
        _db.collection("predictions").add({
            "uid": uid,
            "email": email,
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "timestamp": firestore.SERVER_TIMESTAMP,
        })
    except Exception:
        pass

def fb_get_predictions(limit=100):
    """Fetch last `limit` predictions."""
    if not _fb_ready or _db is None:
        return []
    try:
        docs = (_db.collection("predictions")
                .order_by("timestamp", direction="DESCENDING")
                .limit(limit)
                .stream())
        return [d.to_dict() for d in docs]
    except Exception:
        return []

def fb_get_all_users():
    """Fetch all user records."""
    if not _fb_ready or _db is None:
        return []
    try:
        docs = _db.collection("users").stream()
        return [(d.id, d.to_dict()) for d in docs]
    except Exception:
        return []

def fb_set_role(uid: str, role: str):
    """Set a user's role."""
    if not _fb_ready or _db is None:
        return False
    try:
        _db.collection("users").document(uid).update({"role": role})
        return True
    except Exception:
        return False

def is_first_user():
    """Check if there are any users in the system."""
    if not _fb_ready or _db is None:
        return False
    try:
        users = list(_db.collection("users").limit(1).stream())
        return len(users) == 0
    except Exception:
        return True

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroScribe",
    page_icon="NS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: #f5f0e8; }
.main { background: #f5f0e8 !important; }
.block-container { padding: 1.4rem 2rem !important; max-width: 1280px; }

[data-testid="stSidebar"] { background: #1a1510 !important; border-right: 3px solid #c8a96e; }
[data-testid="stSidebar"] * { color: #e8d8b8 !important; }
[data-testid="stSidebar"] .stRadio label { font-family: 'Space Mono', monospace !important; font-size: .78rem !important; }

.hero { background: #1a1510; padding: 2.2rem 2.8rem; margin-bottom: 1.8rem; border-left: 6px solid #c8a96e; }
.hero-title { font-family: 'Bebas Neue', cursive; font-size: 4rem; color: #f0e0c0; letter-spacing: 3px; line-height: 1; margin: 0; }
.hero-title em { color: #c8a96e; font-style: normal; }
.hero-sub { font-family: 'Space Mono', monospace; font-size: .65rem; color: #806040; letter-spacing: .2em; text-transform: uppercase; margin-top: .4rem; }
.hero-badge { background: #c8a96e; color: #1a1510; font-family: 'Space Mono', monospace; font-size: .6rem; font-weight: 700; letter-spacing: .15em; padding: .25rem .7rem; border-radius: 2px; display: inline-block; margin-top: .6rem; }
.admin-badge { background: #c8a96e; color: #1a1510; font-family: 'Space Mono', monospace; font-size: .58rem; font-weight: 700; letter-spacing: .12em; padding: .2rem .55rem; border-radius: 2px; display: inline-block; }
.user-badge  { background: #3a5070; color: #c0d8f0; font-family: 'Space Mono', monospace; font-size: .58rem; font-weight: 700; letter-spacing: .12em; padding: .2rem .55rem; border-radius: 2px; display: inline-block; }

.card { background: #fff; border: 1px solid #e0d5c0; border-radius: 4px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; box-shadow: 2px 2px 0 #e0d5c0; }
.card-dark { background: #1a1510; border: 1px solid #3a2e20; border-radius: 4px; padding: 1.4rem 1.6rem; margin-bottom: 1rem; }

.section-label { font-family: 'Space Mono', monospace; font-size: .65rem; font-weight: 700; letter-spacing: .2em; text-transform: uppercase; color: #806040; margin-bottom: .8rem; display: flex; align-items: center; gap: .6rem; }
.section-label::after { content: ''; flex: 1; height: 1px; background: #d8c8a8; }

.metric-big { font-family: 'Bebas Neue', cursive; font-size: 3.5rem; line-height: 1; color: #1a1510; letter-spacing: 1px; }
.metric-label { font-family: 'Space Mono', monospace; font-size: .6rem; color: #806040; letter-spacing: .15em; text-transform: uppercase; margin-top: .2rem; }

.result-wrap { background: #1a1510; border-left: 6px solid #c8a96e; border-radius: 4px; padding: 2rem; text-align: center; }
.result-emo { font-family: 'Bebas Neue', cursive; font-size: 4.5rem; color: #f0e0c0; letter-spacing: 4px; line-height: 1; }
.result-conf { font-family: 'Space Mono', monospace; font-size: .8rem; color: #c8a96e; letter-spacing: .1em; margin-top: .4rem; }

.bar-wrap { margin: .4rem 0; display: flex; align-items: center; gap: .6rem; }
.bar-label { font-family: 'Space Mono', monospace; font-size: .7rem; color: #4a3820; min-width: 75px; text-transform: capitalize; }
.bar-track { flex: 1; height: 8px; background: #ede5d0; border-radius: 0; overflow: hidden; }
.bar-fill { height: 8px; border-radius: 0; transition: width .4s ease; }
.bar-pct { font-family: 'Space Mono', monospace; font-size: .68rem; color: #806040; min-width: 40px; text-align: right; }

.stButton > button { background: #1a1510 !important; color: #f0e0c0 !important; border: 2px solid #c8a96e !important; border-radius: 2px !important; font-family: 'Space Mono', monospace !important; font-size: .75rem !important; letter-spacing: .1em !important; padding: .55rem 1.6rem !important; text-transform: uppercase !important; transition: all .2s !important; }
.stButton > button:hover { background: #c8a96e !important; color: #1a1510 !important; transform: translateY(-2px) !important; }
.stProgress > div > div > div { background: #c8a96e !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PREPROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def apply_clahe(pil_img: Image.Image) -> Image.Image:
    gray  = np.array(pil_img.convert("L"), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return Image.fromarray(clahe.apply(gray), mode="L")

def clahe_bgr(gray_uint8: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_uint8)

class CLAHETransform:
    def __call__(self, img):
        return apply_clahe(img)

def get_transform(train=False, mean=0.5, std=0.5):
    ops = [CLAHETransform(), transforms.Grayscale(1), transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(.10,.10), shear=8),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1,1.5)),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize((mean,),(std,))]
    if train:
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02,0.12), ratio=(0.3,3.3)))
    return transforms.Compose(ops)

def preprocess_roi_for_inference(roi_bgr, mean, std):
    gray   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray   = clahe_bgr(gray)
    gray   = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(gray).float().unsqueeze(0) / 255.0
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)

def compute_dataset_stats(dataset_path):
    base = transforms.Compose([CLAHETransform(), transforms.Grayscale(1),
                                transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor()])
    try:
        ds = datasets.ImageFolder(dataset_path, transform=base)
        ld = DataLoader(ds, 256, False, num_workers=0)
        ms = ss = n = 0.0
        for imgs,_ in ld:
            ms+=imgs.mean().item(); ss+=imgs.std().item(); n+=1
        return ms/n, max(ss/n, 1e-6)
    except Exception:
        return 0.5, 0.5

# ══════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop=0.05):
        super().__init__()
        self.dw=nn.Conv2d(in_ch,in_ch,3,stride=stride,padding=1,groups=in_ch,bias=False)
        self.pw=nn.Conv2d(in_ch,out_ch,1,bias=False)
        self.bn=nn.BatchNorm2d(out_ch); self.act=nn.ReLU6(inplace=True)
        self.drop=nn.Dropout2d(drop)
    def forward(self,x): return self.drop(self.act(self.bn(self.pw(self.dw(x)))))

class MiniEmotionNet(nn.Module):
    def __init__(self,num_classes=8):
        super().__init__()
        self.entry=nn.Sequential(nn.Conv2d(1,16,3,padding=1,bias=False),nn.BatchNorm2d(16),nn.ReLU6(inplace=True))
        self.body=nn.Sequential(
            DepthwiseSepConv(16,32,stride=2,drop=0.05), DepthwiseSepConv(32,32,drop=0.05),
            DepthwiseSepConv(32,64,stride=2,drop=0.08), DepthwiseSepConv(64,64,drop=0.08),
            DepthwiseSepConv(64,128,stride=2,drop=0.10),DepthwiseSepConv(128,128,drop=0.10),
        )
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.4),
                                nn.Linear(128,64),nn.ReLU6(inplace=True),nn.Dropout(0.2),nn.Linear(64,num_classes))
    def forward(self,x): return self.head(self.body(self.entry(x)))
    def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ══════════════════════════════════════════════════════════════════
#  TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════

def mixup_batch(x,y,alpha=0.4):
    lam=float(np.random.beta(alpha,alpha)); idx=torch.randperm(x.size(0))
    return lam*x+(1-lam)*x[idx],y,y[idx],lam

def mixup_loss(crit,pred,ya,yb,lam): return lam*crit(pred,ya)+(1-lam)*crit(pred,yb)

class EarlyStopping:
    def __init__(self,patience=5,min_delta=1e-4):
        self.patience=patience; self.min_delta=min_delta
        self.best_loss=float('inf'); self.counter=0; self.best_state=None
    def step(self,val_loss,state):
        if val_loss<self.best_loss-self.min_delta:
            self.best_loss=val_loss; self.counter=0
            self.best_state={k:v.clone() for k,v in state.items()}
        else: self.counter+=1
        return self.counter>=self.patience

# ══════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════

def predict_tta(model,img,n_aug=6,mean=0.5,std=0.5):
    clean=get_transform(False,mean,std); aug=get_transform(True,mean,std)
    probs=[]
    with torch.no_grad():
        probs.append(torch.softmax(model(clean(img).unsqueeze(0)),1))
        for _ in range(n_aug-1):
            probs.append(torch.softmax(model(aug(img).unsqueeze(0)),1))
    avg=torch.stack(probs).mean(0).squeeze().numpy()
    return int(np.argmax(avg)),avg

def predict_pil(model,img,mean=0.5,std=0.5,use_tta=False,n_aug=6):
    if use_tta: return predict_tta(model,img,n_aug,mean,std)
    tfm=get_transform(False,mean,std)
    with torch.no_grad():
        probs=torch.softmax(model(tfm(img).unsqueeze(0)),1).squeeze().numpy()
    return int(np.argmax(probs)),probs

# ══════════════════════════════════════════════════════════════════
#  FACE DETECTION HELPERS (IMPROVED)
# ══════════════════════════════════════════════════════════════════

def detect_faces(gray_uint8,cascade):
    eq=cv2.equalizeHist(gray_uint8)
    det=cascade.detectMultiScale(eq,scaleFactor=FACE_SCALE,minNeighbors=FACE_NEIGHBORS,
                                  minSize=(FACE_MIN_SIZE,FACE_MIN_SIZE),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(det)==0: return []
    faces=det.tolist(); faces.sort(key=lambda r:r[2]*r[3],reverse=True)
    return faces

def pad_face_roi(x,y,w,h,fh,fw,pad=FACE_PAD):
    dx,dy=int(w*pad),int(h*pad)
    return max(0,x-dx),max(0,y-dy),min(fw,x+w+dx),min(fh,y+h+dy)

def centre_crop_bgr(frame):
    h,w=frame.shape[:2]; cy,cx=h//2,w//2
    ch,cw=int(h*.6),int(w*.6)
    return frame[max(0,cy-ch//2):min(h,cy+ch//2), max(0,cx-cw//2):min(w,cx+cw//2)]

class ProbSmoother:
    def __init__(self,n_classes,alpha=EMA_ALPHA):
        self.alpha=alpha; self.smoothed=np.ones(n_classes,dtype=np.float32)/n_classes
    def update(self,new_probs):
        if new_probs is not None:
            self.smoothed=self.alpha*new_probs+(1-self.alpha)*self.smoothed
            self.smoothed/=self.smoothed.sum()
        return self.smoothed.copy()
    @property
    def top(self): idx=int(np.argmax(self.smoothed)); return idx,float(self.smoothed[idx])

# ══════════════════════════════════════════════════════════════════
#  IMPROVED FUNCTION: Process face image for inference
# ══════════════════════════════════════════════════════════════════

def process_face_for_inference(face_roi_rgb, mean=0.5, std=0.5):
    """
    CRITICAL FIX: Properly preprocess faces with CLAHE + normalization
    This matches exactly what the model was trained on
    """
    # Convert to grayscale (emotion recognition works better on grayscale)
    gray = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE - THIS WAS MISSING IN ORIGINAL!
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    
    # Resize to model input size
    gray_resized = cv2.resize(gray_clahe, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor and normalize
    tensor = torch.from_numpy(gray_resized).float().unsqueeze(0) / 255.0
    tensor = (tensor - mean) / std
    
    return tensor.unsqueeze(0)

def extract_faces_from_image(image_rgb, cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
    """
    Extract the largest face from an RGB image
    Returns: cropped face ROI or None if no face found
    """
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    faces = detect_faces(gray, face_cascade)
    
    if len(faces) == 0:
        return None
    
    # Get largest face
    x, y, w, h = faces[0]
    
    # Add padding for better face detection
    h_img, w_img = image_rgb.shape[:2]
    x1, y1, x2, y2 = pad_face_roi(x, y, w, h, h_img, w_img, FACE_PAD)
    
    # Extract face ROI
    face_roi = image_rgb[y1:y2, x1:x2]
    
    if face_roi.size == 0:
        return None
        
    return face_roi

# ══════════════════════════════════════════════════════════════════
#  DATASET UTILS
# ══════════════════════════════════════════════════════════════════

def extract_zip_dataset(zb):
    tmp=tempfile.mkdtemp(prefix="ns_")
    try:
        with zipfile.ZipFile(io.BytesIO(zb)) as zf: zf.extractall(tmp)
    except Exception as e: st.error(f"ZIP error: {e}"); return None
    return tmp

def find_split(root,split):
    c=os.path.join(root,split)
    if os.path.isdir(c): return c
    entries=[e for e in os.listdir(root) if os.path.isdir(os.path.join(root,e)) and not e.startswith('.')]
    if len(entries)==1:
        c2=os.path.join(root,entries[0],split)
        if os.path.isdir(c2): return c2
        return os.path.join(root,entries[0])
    return root if os.path.isdir(root) else None

@st.cache_resource(show_spinner=False)
def load_model_cached(path,_mtime):
    m=MiniEmotionNet(num_classes=len(CLASSES))
    try:
        sd=torch.load(path,map_location="cpu",weights_only=False)
        m.load_state_dict(sd["model_state"] if isinstance(sd,dict) and "model_state" in sd else sd,strict=True)
    except Exception as e: st.error(f"Model load error: {e}"); return None
    m.eval(); return m

def load_norm_stats(path):
    try:
        sd=torch.load(path,map_location="cpu",weights_only=False)
        if isinstance(sd,dict) and "norm_mean" in sd: return sd["norm_mean"],sd["norm_std"]
    except: pass
    return 0.5,0.5

# ══════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════════════

def fig_buf(fig):
    buf=io.BytesIO(); fig.savefig(buf,format='png',bbox_inches='tight',dpi=120)
    buf.seek(0); plt.close(fig); return buf

def render_confusion_matrix(all_y,all_p,class_names,title="CM"):
    cm=confusion_matrix(all_y,all_p)
    fig,ax=plt.subplots(figsize=(max(7,len(class_names)),max(5,len(class_names)-1)))
    sns.heatmap(cm,annot=True,fmt='d',cmap='YlOrBr',xticklabels=class_names,
                yticklabels=class_names,linewidths=.5,linecolor='#e8dcc8',annot_kws={"size":9},ax=ax)
    ax.set_xlabel("Predicted",fontsize=9,labelpad=8); ax.set_ylabel("True",fontsize=9,labelpad=8)
    ax.set_title(title,fontsize=11,pad=12)
    plt.xticks(rotation=45,ha='right',fontsize=8); plt.yticks(rotation=0,fontsize=8)
    plt.tight_layout(); return fig_buf(fig)

def render_metrics_table(report,class_names):
    th=("padding:.6rem .7rem;text-align:center;color:#806040;font-family:\"Space Mono\",monospace;"
        "font-weight:700;letter-spacing:.1em;font-size:.65rem;text-transform:uppercase;")
    rows=""
    for cls in class_names:
        if cls not in report: continue
        r=report[cls]
        rows+=f"""<tr style='border-bottom:1px solid #ede5d0;'>
          <td style='padding:.5rem .7rem;'><span style='font-size:1rem;'>{EMOJIS.get(cls,'')}</span>
            <span style='font-family:"Space Mono",monospace;font-size:.75rem;font-weight:700;
              color:{COLORS.get(cls,"#888")};margin-left:.4rem;text-transform:uppercase;'>{cls}</span></td>
          <td style='padding:.5rem .7rem;text-align:center;font-family:"Space Mono",monospace;font-size:.75rem;color:#3a2e20;'>{r['precision']*100:.1f}%</td>
          <td style='padding:.5rem .7rem;text-align:center;font-family:"Space Mono",monospace;font-size:.75rem;color:#3a2e20;'>{r['recall']*100:.1f}%</td>
          <td style='padding:.5rem .7rem;text-align:center;font-family:"Space Mono",monospace;font-size:.75rem;font-weight:700;color:#1a1510;'>{r['f1-score']*100:.1f}%</td>
          <td style='padding:.5rem .7rem;text-align:center;font-family:"Space Mono",monospace;font-size:.75rem;color:#806040;'>{int(r['support'])}</td>
        </tr>"""
    return f"""<div class="card" style='padding:0;overflow-x:auto;'>
    <table style='width:100%;border-collapse:collapse;'>
      <thead><tr style='background:#f0e8d8;'>
        <th style='{th}text-align:left;'>Class</th>
        <th style='{th}'>Precision</th><th style='{th}'>Recall</th>
        <th style='{th}'>F1</th><th style='{th}'>Support</th>
       </tr></thead><tbody>{rows}</tbody></div>"""

# ══════════════════════════════════════════════════════════════════
#  AUTHENTICATION SCREENS
# ══════════════════════════════════════════════════════════════════

def render_auth_screen():
    """Render the authentication screen with Sign Up and Login tabs."""
    
    st.markdown("""
    <div style='text-align:center;margin-bottom:2rem;'>
      <div style='font-family:"Bebas Neue",cursive;font-size:4rem;color:#1a1510;letter-spacing:6px;'>
        Neuro<span style='color:#c8a96e;'>Scribe</span>
      </div>
      <div style='font-family:"Space Mono",monospace;font-size:.7rem;color:#806040;
                  letter-spacing:.3em;text-transform:uppercase;margin-top:.5rem;'>
        Face Emotion Recognition System
      </div>
    </div>""", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 SIGN UP", "📝 LOG IN"])
    
    with tab1:
        with st.form("signup_form"):
            signup_email = st.text_input("Email", placeholder="you@example.com")
            signup_password = st.text_input("Password", placeholder="•••••••• (min. 6 characters)", type="password")
            signup_confirm = st.text_input("Confirm Password", placeholder="••••••••", type="password")
            signup_role = st.selectbox("Role", ["user", "admin"])
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button("Create Account", use_container_width=True)
            
            if submitted:
                if not signup_email or not signup_password:
                    st.error("Please fill in all fields.")
                elif signup_password != signup_confirm:
                    st.error("Passwords do not match.")
                elif len(signup_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    with st.spinner("Creating account..."):
                        _init_firebase()
                        is_first = is_first_user()
                        if is_first:
                            signup_role = "admin"
                            st.info("First user will be created as ADMIN.")
                        
                        user, error = fb_sign_up(signup_email, signup_password, signup_role)
                        
                        if error:
                            st.error(error)
                        else:
                            role = fb_get_role(user["localId"])
                            st.session_state["user"] = user
                            st.session_state["role"] = role
                            st.session_state["email"] = signup_email
                            st.session_state["show_main_app"] = True
                            st.success("✅ Account created successfully!")
                            st.rerun()
    
    with tab2:
        with st.form("login_form"):
            login_email = st.text_input("Email", placeholder="you@example.com")
            login_password = st.text_input("Password", placeholder="••••••••", type="password")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if not login_email or not login_password:
                    st.error("Please enter email and password.")
                else:
                    with st.spinner("Signing in..."):
                        _init_firebase()
                        user, error = fb_sign_in(login_email, login_password)
                        
                        if error:
                            st.error(error)
                        else:
                            role = fb_get_role(user["localId"])
                            st.session_state["user"] = user
                            st.session_state["role"] = role
                            st.session_state["email"] = login_email
                            st.session_state["show_main_app"] = True
                            st.success(f"Welcome back, {login_email}!")
                            st.rerun()

def render_role_selection():
    """Render role selection screen."""
    st.markdown("""
    <div class="hero" style='text-align:center;'>
      <div class="hero-title">Welcome, <em>{}</em>!</div>
      <div class="hero-sub">Your role has been detected</div>
      <div class="hero-badge">{}</div>
    </div>""".format(st.session_state.get("email", "User"), 
                     "⚙ ADMIN MODE" if st.session_state.get("role") == "admin" else "👤 USER MODE"), 
    unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continue to Dashboard →", use_container_width=True):
            st.session_state["show_main_app"] = True
            st.rerun()

# ══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════

def render_main_app():
    """Render the main application interface."""
    
    _current_user = st.session_state["user"]
    _current_role = st.session_state.get("role", "user")
    _current_email = st.session_state.get("email", "")
    _current_uid = _current_user.get("localId", "demo") if isinstance(_current_user, dict) else "demo"
    _is_admin = (_current_role == "admin")
    _page_list = ADMIN_PAGES if _is_admin else USER_PAGES
    
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:1.2rem 0 .6rem;'>
          <div style='font-family:"Bebas Neue",cursive;font-size:1.6rem;color:#c8a96e;letter-spacing:3px;'>NeuroScribe</div>
          <div style='font-family:"Space Mono",monospace;font-size:.58rem;color:#5a4030;letter-spacing:.2em;text-transform:uppercase;margin-top:.2rem;'>
            CPU Edition · v4.2
          </div>
        </div>""", unsafe_allow_html=True)
        
        badge_html = (f'<span class="admin-badge">⚙ ADMIN</span>'
                      if _is_admin else '<span class="user-badge">👤 USER</span>')
        st.markdown(f"""
        <div style='background:#2a1f14;border-radius:4px;padding:.6rem .8rem;margin-bottom:.6rem;'>
          <div style='font-family:"Space Mono",monospace;font-size:.65rem;color:#c8b890;
                      word-break:break-all;'>{_current_email}</div>
          <div style='margin-top:.3rem;'>{badge_html}</div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("---")
        page = st.radio("Navigate", _page_list, label_visibility="collapsed")
        st.markdown("---")
        
        model_path_input = st.text_input("Weights path", value=MODEL_PATH, label_visibility="collapsed")
        
        if st.button("Sign Out", key="signout"):
            for k in ["user","role","email","show_main_app"]:
                st.session_state.pop(k, None)
            st.rerun()
    
    # Hero section
    role_label = "Admin Mode" if _is_admin else "User Mode"
    st.markdown(f"""
    <div class="hero">
      <div>
        <div class="hero-title">Neuro<em>Scribe</em></div>
        <div class="hero-sub">v4.2 · Fixed Live Webcam Detection · CLAHE + Face Detection</div>
        <div class="hero-badge">⚡ {role_label} · {_current_email}</div>
      </div>
    </div>""", unsafe_allow_html=True)
    
    # ==================== PAGE: OVERVIEW ====================
    if page == "Overview":
        c1,c2,c3,c4 = st.columns(4)
        for col,val,lbl,note in [
            (c1,"8","Classes","Emotion types"),
            (c2,"~155K","Params","Lightweight"),
            (c3,"48×48","Input","Grayscale+CLAHE"),
            (c4,"EMA","Smoothing","Flicker-free"),
        ]:
            col.markdown(f'<div class="card" style="text-align:center;"><div class="metric-big">{val}</div><div class="metric-label">{lbl} · {note}</div></div>',unsafe_allow_html=True)
        
        st.markdown('<div class="section-label">Emotion Classes</div>', unsafe_allow_html=True)
        ecols=st.columns(8)
        for col,cls in zip(ecols,CLASSES):
            col.markdown(f"""
            <div style='text-align:center;padding:.8rem .2rem;background:#fff;
                        border:1px solid {COLORS[cls]}40;border-top:3px solid {COLORS[cls]};border-radius:4px;'>
              <div style='font-size:1.8rem;'>{EMOJIS[cls]}</div>
              <div style='font-size:.65rem;color:{COLORS[cls]};font-family:"Space Mono",monospace;
                          font-weight:700;margin-top:.3rem;text-transform:uppercase;'>{cls}</div>
            </div>""", unsafe_allow_html=True)
        
        if _is_admin:
            st.markdown('<div class="section-label" style="margin-top:1rem;">Admin Quick Start</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card-dark">
              <div style='font-size:.8rem;color:#c8b890;line-height:2;'>
                <b style='color:#f0d080;'>Step 1</b> → Dataset Upload → upload ZIP<br>
                <b style='color:#f0d080;'>Step 2</b> → Train → configure and run<br>
                <b style='color:#f0d080;'>Step 3</b> → Inference / Live Webcam → test<br>
                <b style='color:#f0d080;'>Step 4</b> → Admin Panel → manage users &amp; view logs
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-label" style="margin-top:1rem;">User Quick Start</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card-dark">
              <div style='font-size:.8rem;color:#c8b890;line-height:2;'>
                <b style='color:#f0d080;'>Inference</b> → upload a face photo → get emotion<br>
                <b style='color:#f0d080;'>Live Webcam</b> → real-time detection with face cropping + CLAHE<br>
                Your predictions are logged automatically.
              </div>
            </div>""", unsafe_allow_html=True)
    
    # ==================== PAGE: DATASET UPLOAD ====================
    elif page == "Dataset Upload" and _is_admin:
        st.markdown('<div class="section-label">Upload ZIP Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
          <pre style='background:#f0e8d8;border:1px solid #d8c8a0;border-radius:4px;
                      padding:.8rem 1rem;font-size:.75rem;color:#3a2e20;line-height:1.7;'>
dataset.zip
├── train/
│   ├── angry/ contempt/ disgust/ fear/
│   └── happy/ neutral/ sad/ surprise/
└── val/  (same structure — optional)
          </pre>
        </div>""", unsafe_allow_html=True)
        
        uz=st.file_uploader("Upload dataset ZIP",type=["zip"])
        if uz:
            with st.spinner("Extracting..."):
                root=extract_zip_dataset(uz.read())
            if root:
                st.session_state["dataset_root"]=root
                tp=find_split(root,"train"); vp=find_split(root,"val")
                if tp:
                    with st.spinner("Computing CLAHE-matched stats..."):
                        m_,s_=compute_dataset_stats(tp)
                    st.session_state["norm_mean"]=m_; st.session_state["norm_std"]=s_
                    st.success(f"✅ Mean: **{m_:.4f}** · Std: **{s_:.4f}**")
                summary=""
                for sn,sp in [("train",tp),("val",vp)]:
                    if sp and os.path.isdir(sp):
                        found=sorted([d for d in os.listdir(sp) if os.path.isdir(os.path.join(sp,d)) and not d.startswith('.')])
                        counts={c:len([f for f in os.listdir(os.path.join(sp,c)) if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.webp'))]) for c in found if os.path.isdir(os.path.join(sp,c))}
                        summary+=f"<div style='padding:.5rem 0;border-bottom:1px solid #ede5d0;'><span style='font-family:\"Space Mono\",monospace;font-size:.7rem;font-weight:700;color:#c8a96e;text-transform:uppercase;'>{sn}</span><span style='font-size:.75rem;color:#3a2e20;margin-left:1rem;'>{sum(counts.values()):,} images · {len(found)} classes</span><div style='font-size:.72rem;color:#806040;margin-top:.3rem;'>{' · '.join(f'{c}:{counts[c]}' for c in found)}</div></div>"
                st.markdown(f'<div class="card">{summary}</div>',unsafe_allow_html=True)
        else:
            st.info("Using previously uploaded dataset." if "dataset_root" in st.session_state else "No dataset uploaded yet.")
    
    # ==================== PAGE: TRAIN ====================
    elif page == "Train" and _is_admin:
        st.markdown('<div class="section-label">Training Configuration</div>', unsafe_allow_html=True)
        root=st.session_state.get("dataset_root")
        if not root: 
            st.warning("⚠ Upload a dataset first (go to Dataset Upload page).")
        else:
            tp=find_split(root,"train"); vp=find_split(root,"val")
            if not tp or not os.path.isdir(tp): 
                st.error("No train/ folder found.")
            else:
                st.info(f"Train: `{tp}` | Val: `{vp or 'auto 10%'}`")
                c1,c2,c3,c4=st.columns(4)
                epochs=c1.number_input("Epochs",1,200,20)
                bs=c2.number_input("Batch Size",8,256,32,step=8)
                lr=c3.number_input("Learning Rate",1e-5,1e-1,1e-3,format="%.5f")
                workers=c4.number_input("Workers",0,4,0)
                c5,c6,c7=st.columns(3)
                use_mx=c5.checkbox("Mixup",value=True)
                use_es=c6.checkbox("Early Stopping",value=True)
                es_p=c7.number_input("ES Patience",2,20,5)
                cm_freq=st.selectbox("Confusion matrix every N epochs",[1,2,5,10],index=1)
                
                if st.button("> Start Training", key="train_btn"):
                    try:
                        nm=st.session_state.get("norm_mean",0.5); ns=st.session_state.get("norm_std",0.5)
                        st.info(f"CLAHE stats — Mean:{nm:.4f} Std:{ns:.4f}")
                        t_tfm=get_transform(True,nm,ns); v_tfm=get_transform(False,nm,ns)
                        with st.spinner("Loading dataset..."):
                            fds=datasets.ImageFolder(tp,transform=t_tfm)
                            nc=len(fds.classes); cn=fds.classes
                            if vp and os.path.isdir(vp):
                                vds=datasets.ImageFolder(vp,transform=v_tfm); tds=fds
                            else:
                                nv=max(1,int(len(fds)*.1)); nt=len(fds)-nv
                                tds,vds=torch.utils.data.random_split(fds,[nt,nv])
                            tld=DataLoader(tds,int(bs),True,num_workers=int(workers))
                            vld=DataLoader(vds,int(bs),False,num_workers=int(workers))
                        try:
                            lbls=[tds.dataset.targets[i] for i in tds.indices] if hasattr(tds,'dataset') else fds.targets
                            cnt=np.bincount(lbls,minlength=nc).astype(float)
                            w=torch.tensor(1.0/(cnt+1e-6),dtype=torch.float32); w=w/w.sum()*nc
                        except: w=None
                        crit=nn.CrossEntropyLoss(weight=w,label_smoothing=0.1)
                        mdl=MiniEmotionNet(num_classes=nc)
                        opt=optim.Adam(mdl.parameters(),lr=float(lr),weight_decay=1e-4)
                        sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=int(epochs))
                        es=EarlyStopping(int(es_p)) if use_es else None
                        st.success(f"{len(tds)} train / {len(vds)} val · {nc} classes")
                        hist={'tl':[],'vl':[],'ta':[],'va':[]}
                        prog=st.progress(0); stat_ph=st.empty(); chart_ph=st.empty()
                        tbl_ph=st.empty(); cm_ttl=st.empty(); cm_ph=st.empty()
                        
                        for ep in range(1,int(epochs)+1):
                            mdl.train(); tl=tc=tt=0
                            for bi,(x,y) in enumerate(tld):
                                opt.zero_grad()
                                if use_mx and np.random.random()>0.5:
                                    mx,ya,yb,lam=mixup_batch(x,y); out=mdl(mx)
                                    loss=mixup_loss(crit,out,ya,yb,lam)
                                    p=out.argmax(1); tc+=(lam*p.eq(ya).float()+(1-lam)*p.eq(yb).float()).sum().item()
                                else:
                                    out=mdl(x); loss=crit(out,y); tc+=out.argmax(1).eq(y).sum().item()
                                loss.backward(); opt.step()
                                tl+=loss.item()*x.size(0); tt+=y.size(0)
                            mdl.eval(); vl=vc=vt=0; avp=[]; avy=[]
                            with torch.no_grad():
                                for x,y in vld:
                                    out=mdl(x); loss=crit(out,y); vl+=loss.item()*x.size(0)
                                    p=out.argmax(1); vc+=p.eq(y).sum().item(); vt+=y.size(0)
                                    avp.extend(p.numpy()); avy.extend(y.numpy())
                            sched.step()
                            ta_,va_=100*tc/tt,100*vc/vt; tl_,vl_=tl/tt,vl/vt
                            for k,v_ in [('tl',tl_),('vl',vl_),('ta',ta_),('va',va_)]: hist[k].append(v_)
                            prog.progress(ep/int(epochs))
                            stat_ph.markdown(f"""<div style='display:flex;gap:.8rem;flex-wrap:wrap;margin:.3rem 0;'>
                              <span style='background:#1a1510;color:#f0e0c0;font-family:"Space Mono",monospace;font-size:.7rem;padding:.25rem .7rem;border-radius:2px;'>Epoch {ep}/{int(epochs)}</span>
                              <span style='background:#c8f0d0;color:#2a5030;font-family:"Space Mono",monospace;font-size:.7rem;padding:.25rem .7rem;border-radius:2px;'>Train {ta_:.1f}%</span>
                              <span style='background:#c8a96e30;color:#6a4010;font-family:"Space Mono",monospace;font-size:.7rem;padding:.25rem .7rem;border-radius:2px;'>Val {va_:.1f}%</span>
                              <span style='background:#f0f0f0;color:#606060;font-family:"Space Mono",monospace;font-size:.7rem;padding:.25rem .7rem;border-radius:2px;'>Loss {vl_:.4f}</span>
                            </div>""",unsafe_allow_html=True)
                            if ep%2==0 or ep==int(epochs):
                                fig,axes=plt.subplots(1,2,figsize=(11,3.5)); rng=range(1,ep+1)
                                axes[0].plot(rng,hist['tl'],color='#1a1510',label='Train',lw=2)
                                axes[0].plot(rng,hist['vl'],color='#c8a96e',label='Val',lw=2,linestyle='--')
                                axes[0].set_title('Loss',fontsize=9); axes[0].legend(); axes[0].grid(True)
                                axes[1].plot(rng,hist['ta'],color='#1a1510',label='Train',lw=2)
                                axes[1].plot(rng,hist['va'],color='#c8a96e',label='Val',lw=2,linestyle='--')
                                axes[1].set_title('Accuracy (%)',fontsize=9); axes[1].legend(); axes[1].grid(True)
                                plt.tight_layout(); chart_ph.image(fig_buf(fig),use_container_width=True)
                            rpt=classification_report(avy,avp,target_names=cn,output_dict=True,zero_division=0)
                            tbl_ph.markdown(render_metrics_table(rpt,cn),unsafe_allow_html=True)
                            if ep%int(cm_freq)==0 or ep==int(epochs):
                                cm_ttl.markdown(f'<div class="section-label">CM — Epoch {ep}</div>',unsafe_allow_html=True)
                                cm_ph.image(render_confusion_matrix(avy,avp,cn,f"Val CM Epoch {ep}"),use_container_width=True)
                            if es and es.step(vl_,mdl.state_dict()):
                                st.warning(f"⏹ Early stopping at epoch {ep}"); mdl.load_state_dict(es.best_state); break
                        
                        torch.save({"model_state":mdl.state_dict(),"norm_mean":nm,"norm_std":ns,"classes":cn},model_path_input)
                        st.success(f"✅ Saved to `{model_path_input}` · Best val: {max(hist['va']):.2f}%")
                    except Exception as e: st.error(f"Training failed: {e}"); st.exception(e)
    
    # ==================== PAGE: INFERENCE ====================
    elif page == "Inference":
        st.markdown('<div class="section-label">Single Image Prediction</div>', unsafe_allow_html=True)
        mdl=nm=ns=None
        if os.path.exists(model_path_input):
            mdl=load_model_cached(model_path_input,os.path.getmtime(model_path_input))
            nm,ns=load_norm_stats(model_path_input)
            if mdl: st.success(f"Model loaded · {mdl.count_params():,} params · mean={nm:.4f} std={ns:.4f}")
        else:
            uw=st.file_uploader("Upload .pth weights",type=["pth"])
            if uw:
                tp2="tmp_inf.pth"
                with open(tp2,"wb") as f: f.write(uw.read())
                mdl=load_model_cached(tp2,time.time()); nm,ns=load_norm_stats(tp2)
        if not mdl: 
            st.warning("⚠ No model loaded. Please upload a trained model file (.pth)")
        else:
            nm=nm or 0.5; ns=ns or 0.5
            use_tta=st.checkbox("Enable TTA",value=True)
            n_aug=st.slider("TTA passes",3,10,6,disabled=not use_tta)
            left,right=st.columns(2,gap="large")
            with left:
                upl=st.file_uploader("Upload Image",type=["jpg","jpeg","png","bmp","webp"],
                                     label_visibility="collapsed",key="inf_img")
                if upl:
                    img=Image.open(upl).convert("RGB")
                    c1_,c2_=st.columns(2)
                    c1_.image(img,caption="Original",use_container_width=True)
                    c2_.image(apply_clahe(img),caption="After CLAHE",use_container_width=True)
            with right:
                if upl:
                    with st.spinner("Predicting..."):
                        idx,probs=predict_pil(mdl,img,nm,ns,use_tta,n_aug)
                    cls=CLASSES[idx] if idx<len(CLASSES) else f"class_{idx}"
                    conf=probs[idx]*100
                    fb_log_prediction(_current_uid,_current_email,cls,conf)
                    st.markdown(f"""
                    <div class="result-wrap">
                      <div style='font-size:3.5rem;'>{EMOJIS.get(cls,"")}</div>
                      <div class="result-emo" style='color:{COLORS.get(cls,"#888")};'>{cls.upper()}</div>
                      <div class="result-conf">{conf:.1f}% {"· TTA ×"+str(n_aug) if use_tta else "· single pass"}</div>
                    </div>""",unsafe_allow_html=True)
                    st.markdown('<div class="section-label" style="margin-top:1rem;">All Probabilities</div>',unsafe_allow_html=True)
                    bars=""
                    for i in np.argsort(probs)[::-1]:
                        c2_=CLASSES[i] if i<len(CLASSES) else f"class_{i}"
                        p=probs[i]*100
                        bars+=f'<div class="bar-wrap"><span class="bar-label">{EMOJIS.get(c2_,"")} {c2_}</span><div class="bar-track"><div class="bar-fill" style="width:{p:.1f}%;background:{COLORS.get(c2_,"#888")};"></div></div><span class="bar-pct">{p:.1f}%</span></div>'
                    st.markdown(f'<div class="card" style="padding:1rem 1.2rem;">{bars}</div>',unsafe_allow_html=True)
    
    # ==================== PAGE: LIVE WEBCAM (FULLY FIXED) ====================
    elif page == "Live Webcam":
        st.markdown('<div class="section-label">Live Webcam — Real-time Detection (FIXED)</div>', unsafe_allow_html=True)
        
        # Display fix information
        st.info("🔧 **FIXES APPLIED:** Face detection cropping, CLAHE preprocessing, proper normalization, TTA enabled")
        
        mdl = None
        nm = ns = 0.5
        if os.path.exists(model_path_input):
            mdl = load_model_cached(model_path_input, os.path.getmtime(model_path_input))
            nm, ns = load_norm_stats(model_path_input)
        
        if not mdl:
            st.warning("⚠ No model loaded. Please train or upload a model first.")
        else:
            st.success(f"✅ Model loaded (mean={nm:.3f}, std={ns:.3f})")
            
            # Enable TTA by default for better accuracy
            use_tta = st.checkbox("Enable TTA (more accurate, slightly slower)", value=True)
            n_aug = st.slider("TTA augmentations", 3, 10, 5, disabled=not use_tta, 
                             help="More augmentations = higher accuracy but slower")
            
            # Camera input
            cam = st.camera_input("📸 Take a photo with a clear face visible")
            
            if cam:
                # Load image
                img = Image.open(cam).convert("RGB")
                img_np = np.array(img)
                
                # ⭐ CRITICAL FIX 1: Extract face from image
                with st.spinner("🔍 Detecting face..."):
                    face_roi = extract_faces_from_image(img_np)
                
                if face_roi is None:
                    st.error("❌ No face detected! Please ensure your face is clearly visible and well-lit.")
                    st.markdown("""
                    <div style='background:#fce8e8;border-left:4px solid #d44;padding:1rem;margin-top:1rem;'>
                      <b>💡 Tips for better face detection:</b><br>
                      • Look directly at the camera<br>
                      • Ensure good lighting (avoid shadows)<br>
                      • Remove glasses if they cause glare<br>
                      • Position face in the center of frame<br>
                      • Avoid extreme angles or tilts
                    </div>""", unsafe_allow_html=True)
                else:
                    # ⭐ CRITICAL FIX 2: Apply CLAHE preprocessing
                    with st.spinner("🎨 Applying CLAHE preprocessing..."):
                        # Convert face ROI to PIL for processing
                        face_pil = Image.fromarray(face_roi)
                        
                        # Apply CLAHE (same as training)
                        face_clahe = apply_clahe(face_pil)
                    
                    # ⭐ CRITICAL FIX 3: Run prediction with proper preprocessing
                    with st.spinner("🧠 Analyzing emotion..."):
                        # Use the fixed prediction function
                        idx, probs = predict_pil(mdl, face_clahe, nm, ns, use_tta, n_aug)
                    
                    cls = CLASSES[idx] if idx < len(CLASSES) else f"class_{idx}"
                    conf = probs[idx] * 100
                    
                    # Log to Firebase
                    fb_log_prediction(_current_uid, _current_email, cls, conf)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📸 Original Image with Face Detection**")
                        # Draw rectangle on original image to show detected face
                        img_with_box = img_np.copy()
                        # Re-detect for visualization
                        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=FACE_SCALE, minNeighbors=FACE_NEIGHBORS,
                                                              minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE))
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (200, 100, 80), 3)
                        st.image(img_with_box, caption="Detected Face Region", use_container_width=True)
                        
                        # Show preprocessed face
                        st.markdown("**🎨 Preprocessed Face (CLAHE)**")
                        st.image(face_clahe, caption="After CLAHE enhancement", use_container_width=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="result-wrap">
                          <div style='font-size:4rem;'>{EMOJIS.get(cls, "")}</div>
                          <div class="result-emo" style='color:{COLORS.get(cls, "#888")};font-size:3rem;'>{cls.upper()}</div>
                          <div class="result-conf" style='font-size:1.2rem;'>{conf:.1f}% confidence</div>
                          <div style='margin-top:1rem;font-size:0.7rem;color:#c8a96e;'>
                            {'✓ TTA enabled (' + str(n_aug) + ' passes)' if use_tta else 'TTA disabled'}
                          </div>
                        </div>""", unsafe_allow_html=True)
                    
                    # Show all probabilities
                    st.markdown('<div class="section-label" style="margin-top:1rem;">All Probabilities</div>', unsafe_allow_html=True)
                    bars = ""
                    for i in np.argsort(probs)[::-1]:
                        c2_ = CLASSES[i] if i < len(CLASSES) else f"class_{i}"
                        p = probs[i] * 100
                        # Only show if probability > 1%
                        if p > 1:
                            bars += f'<div class="bar-wrap"><span class="bar-label">{EMOJIS.get(c2_,"")} {c2_}</span><div class="bar-track"><div class="bar-fill" style="width:{p:.1f}%;background:{COLORS.get(c2_,"#888")};"></div></div><span class="bar-pct">{p:.1f}%</span></div>'
                    st.markdown(f'<div class="card" style="padding:1rem 1.2rem;">{bars}</div>', unsafe_allow_html=True)
                    
                    # Add confidence indicator
                    if conf > 80:
                        st.success("🎯 High confidence prediction!")
                    elif conf > 60:
                        st.info("👍 Moderate confidence prediction")
                    else:
                        st.warning("⚠️ Low confidence. Try improving lighting or face visibility.")
    
    # ==================== PAGE: TEST IMAGES ====================
    elif page == "Test Images" and _is_admin:
        st.markdown('<div class="section-label">Test Images</div>', unsafe_allow_html=True)
        mdl=None; nm=ns=0.5
        if os.path.exists(model_path_input):
            mdl=load_model_cached(model_path_input,os.path.getmtime(model_path_input))
            nm,ns=load_norm_stats(model_path_input)
            if mdl: st.success(f"Model loaded · {mdl.count_params():,} params")
        if not mdl:
            uw=st.file_uploader("Upload .pth weights",type=["pth"], key="test_upload")
            if uw:
                tp2="tmp_test.pth"
                with open(tp2,"wb") as f: f.write(uw.read())
                mdl=load_model_cached(tp2,time.time()); nm,ns=load_norm_stats(tp2)
        if not mdl: 
            st.warning("⚠ No model loaded.")
        else:
            nm=nm or 0.5; ns=ns or 0.5
            t1,t2=st.tabs(["Test ZIP","Individual Images"])
            with t1:
                tz=st.file_uploader("Test ZIP",type=["zip"],key="tz")
                utt=st.checkbox("TTA",value=False,key="tta_t")
                if tz and st.button("> Evaluate",key="ev"):
                    tr=extract_zip_dataset(tz.read())
                    if tr:
                        td=find_split(tr,"test") or tr
                        if os.path.isdir(td):
                            tfm=get_transform(False,nm,ns); ds=datasets.ImageFolder(td,transform=tfm)
                            dl=DataLoader(ds,64,False,num_workers=0); ap,ay=[],[]
                            pg=st.progress(0); mdl.eval()
                            if utt:
                                for i,(p,l) in enumerate(ds.imgs):
                                    img=Image.open(p).convert("RGB"); ix,_=predict_pil(mdl,img,nm,ns,True,5)
                                    ap.append(ix); ay.append(l)
                                    if i%50==0: pg.progress(min(1.,i/len(ds.imgs)))
                            else:
                                with torch.no_grad():
                                    for i,(x,y) in enumerate(dl):
                                        out=mdl(x); ap.extend(out.argmax(1).numpy()); ay.extend(y.numpy())
                                        pg.progress((i+1)/len(dl))
                            pg.empty()
                            rpt=classification_report(ay,ap,target_names=ds.classes,output_dict=True,zero_division=0)
                            acc=rpt['accuracy']*100
                            m1,m2,m3,m4=st.columns(4)
                            for col,nm_,vl in [(m1,"Accuracy",f"{acc:.2f}%"),(m2,"Macro F1",f"{rpt['macro avg']['f1-score']*100:.2f}%"),
                                                (m3,"Wtd F1",f"{rpt['weighted avg']['f1-score']*100:.2f}%"),(m4,"Images",f"{len(ds):,}")]:
                                col.markdown(f'<div class="card" style="text-align:center;"><div class="metric-big">{vl}</div><div class="metric-label">{nm_}</div></div>',unsafe_allow_html=True)
                            st.markdown(render_metrics_table(rpt,ds.classes),unsafe_allow_html=True)
                            st.image(render_confusion_matrix(ay,ap,ds.classes,"Test CM"),use_container_width=True)
            with t2:
                uis=st.file_uploader("Images",type=["jpg","jpeg","png","bmp","webp"],accept_multiple_files=True,key="mi")
                tl2=st.selectbox("True label",["— unknown —"]+CLASSES); ut=st.checkbox("TTA",value=True,key="tta_i")
                if uis and st.button("> Predict All",key="pa"):
                    ok=wr=0; ul=tl2!="— unknown —"
                    for rs in range(0,len(uis),4):
                        row=uis[rs:rs+4]; cols=st.columns(4)
                        for col,u in zip(cols,row):
                            img=Image.open(u).convert("RGB"); ix,pr=predict_pil(mdl,img,nm,ns,ut,5)
                            c_=CLASSES[ix] if ix<len(CLASSES) else f"class_{ix}"
                            ic=ul and c_==tl2; iw=ul and c_!=tl2
                            if ic: ok+=1
                            if iw: wr+=1
                            bdg=(f'<span style="background:#e8f8e8;color:#2a6030;border:1px solid #80c080;font-family:\'Space Mono\',monospace;font-size:.62rem;padding:.18rem .5rem;border-radius:2px;">✓ CORRECT</span>'
                                 if ic else f'<span style="background:#fce8e8;color:#802020;border:1px solid #e08080;font-family:\'Space Mono\',monospace;font-size:.62rem;padding:.18rem .5rem;border-radius:2px;">✗ {c_.upper()}</span>') if ul else ""
                            col.image(img,use_container_width=True)
                            col.markdown(f'<div style="text-align:center;"><div style="font-family:\'Bebas Neue\',cursive;font-size:1.4rem;color:{COLORS.get(c_,"#888")};letter-spacing:2px;">{EMOJIS.get(c_,"")} {c_.upper()}</div><div style="font-family:\'Space Mono\',monospace;font-size:.65rem;color:#806040;">{pr[ix]*100:.1f}%</div>{bdg}</div>',unsafe_allow_html=True)
                    if ul and ok+wr:
                        s1,s2,s3=st.columns(3)
                        for col,nm2,vl,bg,fg in[(s1,"Accuracy",f"{100*ok/(ok+wr):.1f}%","#e8f8e8","#2a5030"),
                                                 (s2,"Correct",f"{ok}/{ok+wr}","#e8f8e8","#2a5030"),
                                                 (s3,"Wrong",f"{wr}/{ok+wr}","#fce8e8","#802020")]:
                            col.markdown(f'<div style="background:{bg};border:1px solid {fg}40;border-radius:4px;padding:1rem;text-align:center;"><div style="font-family:\'Bebas Neue\',cursive;font-size:2.5rem;color:{fg};">{vl}</div><div style="font-family:\'Space Mono\',monospace;font-size:.62rem;color:{fg};">{nm2}</div></div>',unsafe_allow_html=True)
    
    # ==================== PAGE: EVALUATION ====================
    elif page == "Evaluation" and _is_admin:
        st.markdown('<div class="section-label">Model Evaluation</div>', unsafe_allow_html=True)
        root=st.session_state.get("dataset_root")
        td_def=find_split(root,"test") if root else ""
        td=st.text_input("Test directory",value=td_def or "")
        if st.button("> Run Evaluation", key="eval_btn"):
            if not os.path.exists(model_path_input): 
                st.error("No model found. Train or upload a model first.")
            elif not td or not os.path.isdir(td): 
                st.error("Test directory not found.")
            else:
                em=load_model_cached(model_path_input,os.path.getmtime(model_path_input))
                nm2,ns2=load_norm_stats(model_path_input)
                if em:
                    tfm=get_transform(False,nm2,ns2); ds=datasets.ImageFolder(td,transform=tfm)
                    dl=DataLoader(ds,64,False,num_workers=0)
                    st.info(f"{len(ds)} images · mean={nm2:.4f} std={ns2:.4f}")
                    ap,ay=[]; pg=st.progress(0); em.eval()
                    with torch.no_grad():
                        for i,(x,y) in enumerate(dl):
                            out=em(x); ap.extend(out.argmax(1).numpy()); ay.extend(y.numpy())
                            pg.progress((i+1)/len(dl))
                    pg.empty()
                    rpt=classification_report(ay,ap,target_names=ds.classes,output_dict=True,zero_division=0)
                    acc=rpt['accuracy']*100
                    m1,m2,m3=st.columns(3)
                    for col,nm3,vl in[(m1,"Accuracy",f"{acc:.2f}%"),(m2,"Macro F1",f"{rpt['macro avg']['f1-score']*100:.2f}%"),
                                       (m3,"Wtd F1",f"{rpt['weighted avg']['f1-score']*100:.2f}%")]:
                        col.markdown(f'<div class="card" style="text-align:center;"><div class="metric-big">{vl}</div><div class="metric-label">{nm3}</div></div>',unsafe_allow_html=True)
                    st.markdown(render_metrics_table(rpt,ds.classes),unsafe_allow_html=True)
                    st.image(render_confusion_matrix(ay,ap,ds.classes,"Eval CM"),use_container_width=True)
    
    # ==================== PAGE: ADMIN PANEL ====================
    elif page == "Admin Panel" and _is_admin:
        st.markdown('<div class="section-label">Admin Panel</div>', unsafe_allow_html=True)
        fb_ok = _init_firebase()
        if not fb_ok:
            st.warning("⚠ Firebase not configured. Admin features require Firebase setup.")
        else:
            tab_users, tab_logs = st.tabs(["👥 User Management", "📊 Prediction Log"])
            
            with tab_users:
                st.markdown('<div class="section-label">All Users</div>', unsafe_allow_html=True)
                users = fb_get_all_users()
                if not users:
                    st.info("No users found.")
                else:
                    for uid, data in users:
                        role = data.get("role", "user")
                        email = data.get("email", uid)
                        badge = '⚙ ADMIN' if role == "admin" else '👤 USER'
                        bg = '#fff8ec' if role == "admin" else '#f0f4ff'
                        col1, col2 = st.columns([3, 1])
                        col1.markdown(f"""
                        <div style='background:{bg};border:1px solid #e0d5c0;border-radius:4px;padding:.6rem .9rem;'>
                          <div style='font-family:"Space Mono",monospace;font-size:.72rem;color:#1a1510;'>{email}</div>
                          <div style='font-family:"Space Mono",monospace;font-size:.6rem;color:#806040;margin-top:.2rem;'>{badge}</div>
                        </div>""", unsafe_allow_html=True)
                        if uid != _current_uid:
                            new_role = "user" if role == "admin" else "admin"
                            if col2.button(f"→ Make {'Admin' if new_role=='admin' else 'User'}", key=f"role_{uid}"):
                                if fb_set_role(uid, new_role):
                                    st.success(f"Updated {email} → {new_role}")
                                    st.rerun()
            
            with tab_logs:
                st.markdown('<div class="section-label">Recent Predictions</div>', unsafe_allow_html=True)
                preds = fb_get_predictions(50)
                if not preds:
                    st.info("No predictions logged yet.")
                else:
                    emotion_counts = Counter(p.get("emotion","?") for p in preds)
                    st.markdown('<div class="section-label">Emotion Distribution</div>', unsafe_allow_html=True)
                    ecols = st.columns(len(CLASSES))
                    for col, cls in zip(ecols, CLASSES):
                        cnt = emotion_counts.get(cls, 0)
                        col.markdown(f"""
                        <div style='text-align:center;padding:.6rem .2rem;background:#fff;
                                    border-top:3px solid {COLORS[cls]};border:1px solid {COLORS[cls]}40;border-radius:4px;'>
                          <div style='font-size:1.4rem;'>{EMOJIS[cls]}</div>
                          <div style='font-family:"Bebas Neue",cursive;font-size:1.6rem;color:{COLORS[cls]};'>{cnt}</div>
                          <div style='font-family:"Space Mono",monospace;font-size:.58rem;color:#806040;'>{cls}</div>
                        </div>""", unsafe_allow_html=True)
                    
                    st.markdown('<div class="section-label" style="margin-top:1rem;">Log</div>', unsafe_allow_html=True)
                    for p in preds[:20]:
                        em = p.get("emotion", "?")
                        conf = p.get("confidence", 0)
                        mail = p.get("email", "?")
                        ts = str(p.get("timestamp", ""))[:19]
                        st.markdown(f"""
                        <div style='background:#f5f0e8;border-bottom:1px solid #e0d5c0;padding:.5rem 0;font-family:"Space Mono",monospace;font-size:.7rem;'>
                          <span style='color:#806040;'>{ts}</span> | 
                          <span style='color:#3a2e20;'>{mail}</span> → 
                          <span style='color:{COLORS.get(em,"#888")};font-weight:700;'>{em.upper()} {conf:.1f}%</span>
                        </div>""", unsafe_allow_html=True)
                    
                    buf = io.StringIO()
                    writer = csv.DictWriter(buf, fieldnames=["timestamp","email","emotion","confidence","uid"])
                    writer.writeheader()
                    for p in preds:
                        writer.writerow({k: p.get(k, "") for k in ["timestamp","email","emotion","confidence","uid"]})
                    st.download_button("⬇ Download CSV", buf.getvalue(),
                                       file_name="neuroscribe_predictions.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════
#  MAIN CONTROLLER
# ══════════════════════════════════════════════════════════════════

def main():
    """Main application controller"""
    if "user" not in st.session_state:
        render_auth_screen()
    elif not st.session_state.get("show_main_app", False):
        render_role_selection()
    else:
        render_main_app()

if __name__ == "__main__":
    main()