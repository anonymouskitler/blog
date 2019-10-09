# A simple starter kit for Eleventy

Hylia is a lightweight [Eleventy](https://11ty.io) starter kit with [Netlify CMS](https://www.netlifycms.org/) pre-configured, so that you can one-click install a progressive, accessible blog in minutes. It also gives you a well organised starting point to extend it for yourself.

Get started now by **[deploying Hylia to Netlify.][deploy-to-netlify]**

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)][deploy-to-netlify]

<img src="https://hankchizljaw.imgix.net/hylia-github.jpg?auto=format&q=60" width="550" />

## Features

Hylia version 0.4.4 features:

✍️ A pre-configured [Netlify CMS](https://www.netlifycms.org/) setup  
🎨 Customisable design tokens to make it your own  
🌍 Customisable global data and navigation  
📂 Tags and tag archives  
✅ Progressively enhanced, semantic and accessible  
🎈 _Super_ lightweight front-end  
🚰 Sass powered CSS system with utility class generator  
⚙️ Service worker that caches pages so people can read your articles offline  
🚀 An RSS feed for your posts

## Roadmap

💬 [Netlify Forms](https://www.netlify.com/docs/form-handling/) powered comments  
💡 ~~Dark/Light mode toggle~~ [Added in 0.4.0](https://github.com/andybelldesign/hylia/releases/tag/0.4.0)  
🗣 Webmentions  
📖 Pagination  
🐦 Web sharing API integration  
🗒 Offline mode with links to cached pages  
📄 Documentation site  
💅 Proper Sass documentation  
✍️ Proper CMS documentation  
🖼 A facility for you to be able to add your logo / branding

---

## Getting started


You can [deploy Hylia to Netlify with one click][deploy-to-netlify] and you’ll be up and running in minutes!

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)][deploy-to-netlify]

## Terminal commands

### Serve the site locally

```bash
npm start
```

### Build a production version of the site

```bash
npm run production
```

### Compile Sass

```bash
npm run sass:process
```

### Re-generate design tokens for Sass

```bash
npm run sass:tokens
```

## Getting started with the CMS

Before you can use the CMS, you need to do some config in Netlify. Luckily they provide a [very handy guide to get started](https://www.netlify.com/docs/identity/).

In short, though:

- Once you’ve set up the site on Netlify, go to “Settings” > “Identity” and enable Identity
- Scroll down to the “Git Gateway” area, click “Enable Git Gateway” and follow the steps
- Click the “Identity” tab at the top
- Once you’ve enabled identity, click “Invite Users”
- Check the invite link in your inbox and click the link in the email that’s sent to you
- Set a password in the popup box
- Go to `/admin` on your site and login
- You’re in and ready to edit your content!

## Design Tokens and Styleguide

### Design Tokens

Although Hylia has a pretty simple design, you can configure the core design tokens that control the colours, size ratio and fonts.

---

**Note**: _Credit must be given to the hard work [Jina Anne](https://twitter.com/jina) did in order for the concept of design tokens to even exist. You should watch [this video](https://www.youtube.com/watch?v=wDBEc3dJJV8), then [read this article](https://the-pastry-box-project.net/jina-bolton/2015-march-28) and then sign up for [this course](https://aycl.uie.com/virtual_seminars/design_tokens_scaling_design_with_a_single_source_of_truth) to expand your knowledge._

---

To change the design tokens in the CMS, find the “Globals” in the sidebar then in the presented options, select “Theme Settings”.

To change the design tokens directly, edit [`_src/data/tokens.json`](https://github.com/andybelldesign/hylia/blob/master/src/_data/tokens.json).

The tokens are converted into maps that the Sass uses to compile the front-end CSS, so make sure that you maintain the correct structure of `tokens.json`.

### Styleguide

Your version of Hylia ships with a Styleguide by default. You can see a demo of the Styleguide at <https://hylia.website/styleguide/>.

You can edit the Styleguide by opening [`src/styleguide.njk`](https://github.com/andybelldesign/hylia/blob/master/src/styleguide.njk). If you don’t want the Styleguide, delete that file and the page will vanish.


### Content that you can edit

The basic CMS setup allows you to edit the following:

- **Home page**: Edit the content on your homepage
- **Posts**: Create and edit blog posts
- **Generic pages**: Create generic pages that use a similar layout to posts
- **Global site data**: Various bits of global site data such as your url, title, posts per page and author details
- **Navigation**: Edit your primary navigation items
- **Theme**: Edit the design tokens that power the site’s theme

[deploy-to-netlify]: https://app.netlify.com/start/deploy?repository=https://github.com/chandureddivari/blog&stack=cms
